"""따릉이 재배치를 위한 최소 Gymnasium 환경."""

from __future__ import annotations

from dataclasses import dataclass
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class EnvConfig:
    """환경의 크기, 보상, 수요 패턴을 보관한다."""

    # station_names 개수가 action 개수이며, observation의 재고 값 개수이기도 하다.
    station_names: tuple[str, ...]
    # 각 hour마다 대여소별 대여 수요 샘플링 범위를 저장한다.
    demand_ranges: dict[int, tuple[tuple[int, int], ...]]
    # 각 hour마다 대여소별 반납 수량 샘플링 범위를 저장한다.
    return_ranges: dict[int, tuple[tuple[int, int], ...]]
    # daily profile을 사용할 때 episode 후보 날짜를 저장한다.
    daily_dates: tuple[str, ...] = ()
    # daily profile을 사용할 때 날짜/시간/대여소별 실제 대여 count를 저장한다.
    daily_demand_counts: tuple[tuple[tuple[int, ...], ...], ...] = ()
    # daily profile을 사용할 때 날짜/시간/대여소별 실제 반납 count를 저장한다.
    daily_return_counts: tuple[tuple[tuple[int, ...], ...], ...] = ()
    station_capacity: int = 10
    initial_stock_min: int = 2
    initial_stock_max: int = 8
    truck_capacity: int = 5
    target_stock: int = 5
    episode_steps: int = 24
    unmet_penalty: int = 10
    full_penalty: int = 3
    move_cost: int = 1
    initial_truck_bikes: int = 3
    traffic_enabled: bool = False
    traffic_factors: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """설정값이 환경의 기본 가정을 만족하는지 미리 검증한다."""
        if not self.station_names:
            raise ValueError("station_names must not be empty")
        if self.station_capacity <= 0:
            raise ValueError("station_capacity must be positive")
        if self.truck_capacity < 0:
            raise ValueError("truck_capacity must be zero or positive")
        if not 0 <= self.initial_stock_min <= self.initial_stock_max <= self.station_capacity:
            raise ValueError(
                "initial stock range must satisfy "
                "0 <= initial_stock_min <= initial_stock_max <= station_capacity"
            )
        if not 0 <= self.target_stock <= self.station_capacity:
            raise ValueError("target_stock must be between 0 and station_capacity")
        if not 0 <= self.initial_truck_bikes <= self.truck_capacity:
            raise ValueError("initial_truck_bikes must be between 0 and truck_capacity")
        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        if self.traffic_enabled:
            _validate_traffic_factors(self.traffic_factors, self.episode_steps)
        _validate_hourly_ranges(
            name="demand_ranges",
            ranges=self.demand_ranges,
            episode_steps=self.episode_steps,
            station_count=self.station_count,
        )
        _validate_hourly_ranges(
            name="return_ranges",
            ranges=self.return_ranges,
            episode_steps=self.episode_steps,
            station_count=self.station_count,
        )
        _validate_daily_counts(
            dates=self.daily_dates,
            demand_counts=self.daily_demand_counts,
            return_counts=self.daily_return_counts,
            episode_steps=self.episode_steps,
            station_count=self.station_count,
        )

    @property
    def station_count(self) -> int:
        """대여소 개수를 반환한다."""
        return len(self.station_names)


class DdareungiEnv(gym.Env):
    """3개 대여소와 트럭 1대를 가진 최소 따릉이 MDP 환경."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None) -> None:
        """환경 설정과 seed를 받아 환경을 초기화한다."""
        super().__init__()
        if config is None:
            from ddareungi_rl.config_loader import load_default_config

            config = load_default_config()
        self.config = config
        self.rng = random.Random(seed)

        # action은 "다음에 방문할 대여소 index" 하나로 단순화한다.
        self.action_space = spaces.Discrete(self.config.station_count)

        # observation = 대여소별 재고 + 현재 시간대 예상 수요 + 트럭 위치 + 트럭 적재량 + 현재 시간.
        # DQN이 "어디가 비어 있는가"뿐 아니라 "곧 어디에서 빌리려 하는가"도 볼 수 있게 한다.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.config.station_count * 2 + 3,),
            dtype=np.float32,
        )

        # 아래 네 값이 episode 중 계속 변하는 환경 state이다.
        self.station_bikes = [0 for _ in range(self.config.station_count)]
        self.truck_location = 0
        self.truck_bikes = self.config.initial_truck_bikes
        self.time_step = 0
        self.daily_index = 0
        self.active_date = ""

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """새 episode를 시작하고 observation과 info를 반환한다."""
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.rng.seed(seed)
            self.action_space.seed(seed)
        self.station_bikes = [
            self.rng.randint(
                self.config.initial_stock_min,
                self.config.initial_stock_max,
            )
            for _ in range(self.config.station_count)
        ]
        self.truck_location = 0
        self.truck_bikes = self.config.initial_truck_bikes
        self.time_step = 0
        if self.config.daily_dates:
            forced_daily_index = None if options is None else options.get("daily_index")
            if forced_daily_index is None:
                self.daily_index = self.rng.randrange(len(self.config.daily_dates))
            else:
                self.daily_index = int(forced_daily_index) % len(self.config.daily_dates)
            self.active_date = self.config.daily_dates[self.daily_index]
        else:
            self.daily_index = 0
            self.active_date = ""
        info = self._info(reward=0.0, unmet=0, movement_cost=0.0)
        return self._observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """방문할 대여소 action을 적용하고 다음 상태와 보상을 반환한다."""
        action = int(action)
        if action < 0 or action >= self.config.station_count:
            raise ValueError(f"action must be between 0 and {self.config.station_count - 1}")

        traffic_factor = self.traffic_factor()
        movement_cost = (
            float(self.config.move_cost) * traffic_factor
            if action != self.truck_location
            else 0.0
        )

        # 1. 트럭이 action으로 선택된 대여소로 이동한다.
        self.truck_location = action

        # 2. 방문한 대여소에서 목표 재고에 가까워지도록 싣거나 내린다.
        moved_bikes = self._rebalance(action)

        # 3. 현재 시간대의 대여 수요가 발생하고, 부족분은 unmet_demand가 된다.
        demand = self._current_demand()
        served, unmet = self._apply_demand(demand)

        # 4. 현재 시간대의 반납이 발생하고, 빈 칸이 없으면 rejected_returns가 된다.
        returns = self._current_returns()
        accepted_returns, rejected_returns = self._apply_returns(returns)

        # 5. 학습 신호: 헛걸음, 반납 실패, 이동 비용을 모두 벌점으로 계산한다.
        reward = float(
            -self.config.unmet_penalty * unmet
            - self.config.full_penalty * rejected_returns
            - movement_cost
        )

        self.time_step += 1
        terminated = False
        truncated = self.time_step >= self.config.episode_steps
        info = self._info(
            reward=reward,
            unmet=unmet,
            movement_cost=movement_cost,
            demand=demand,
            returns=returns,
            served=served,
            accepted_returns=accepted_returns,
            rejected_returns=rejected_returns,
            moved_bikes=moved_bikes,
            traffic_factor=traffic_factor,
        )
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str:
        """현재 상태를 한 줄 텍스트로 반환한다."""
        station_text = ", ".join(
            f"{name}:{bikes}"
            for name, bikes in zip(self.config.station_names, self.station_bikes)
        )
        return (
            f"t={self.time_step:02d} | {station_text} | "
            f"truck=({self.config.station_names[self.truck_location]}, {self.truck_bikes})"
        )

    def _observation(self) -> np.ndarray:
        """DQN 입력으로 사용할 정규화 observation vector를 만든다."""
        cfg = self.config
        normalized_expected_demand = [
            min(1.0, demand / cfg.station_capacity)
            for demand in self.expected_demand()
        ]
        values = [
            # 각 대여소 현재 재고: 0.0은 비어 있음, 1.0은 가득 참.
            *(bike / cfg.station_capacity for bike in self.station_bikes),
            # 현재 시간대 예상 대여 수요: DQN이 수요가 몰리는 대여소를 직접 볼 수 있게 한다.
            *normalized_expected_demand,
            # 트럭 위치, 적재량, 하루 중 현재 시간도 모두 0~1 범위로 정규화한다.
            self.truck_location / max(1, cfg.station_count - 1),
            self.truck_bikes / cfg.truck_capacity,
            self.time_step / cfg.episode_steps,
        ]
        return np.array(values, dtype=np.float32)

    def traffic_factor(self) -> float:
        """현재 시간대의 트럭 이동 혼잡 계수를 반환한다."""
        if not self.config.traffic_enabled:
            return 1.0
        hour = self.time_step % self.config.episode_steps
        return float(self.config.traffic_factors[hour])

    def _rebalance(self, station_id: int) -> int:
        """방문한 대여소를 target_stock에 가깝게 맞추고 이동 수량을 반환한다."""
        target = self.config.target_stock
        stock = self.station_bikes[station_id]
        if stock < target:
            unload = min(target - stock, self.truck_bikes)
            self.station_bikes[station_id] += unload
            self.truck_bikes -= unload
            return unload
        if stock > target:
            load = min(stock - target, self.config.truck_capacity - self.truck_bikes)
            self.station_bikes[station_id] -= load
            self.truck_bikes += load
            return -load
        return 0

    def _sample(self, ranges: dict[int, tuple[tuple[int, int], ...]]) -> list[int]:
        """현재 시간의 대여소별 수요 또는 반납량을 샘플링한다."""
        hour = self.time_step % self.config.episode_steps
        return [self.rng.randint(low, high) for low, high in ranges[hour]]

    def expected_demand(self) -> list[float]:
        """현재 시간대에 예상되는 대여소별 수요를 반환한다."""
        hour = self.time_step % self.config.episode_steps
        if self.config.daily_demand_counts:
            return [float(value) for value in self.config.daily_demand_counts[self.daily_index][hour]]
        return [
            (low + high) / 2
            for low, high in self.config.demand_ranges[hour]
        ]

    def _current_demand(self) -> list[int]:
        """현재 step에 적용할 대여 count를 반환한다."""
        if self.config.daily_demand_counts:
            hour = self.time_step % self.config.episode_steps
            return list(self.config.daily_demand_counts[self.daily_index][hour])
        return self._sample(self.config.demand_ranges)

    def _current_returns(self) -> list[int]:
        """현재 step에 적용할 반납 count를 반환한다."""
        if self.config.daily_return_counts:
            hour = self.time_step % self.config.episode_steps
            return list(self.config.daily_return_counts[self.daily_index][hour])
        return self._sample(self.config.return_ranges)

    def _apply_demand(self, demand: list[int]) -> tuple[int, int]:
        """수요를 재고에서 차감하고 처리/미처리 수요를 반환한다."""
        served_total = 0
        unmet_total = 0
        for station_id, requested in enumerate(demand):
            served = min(self.station_bikes[station_id], requested)
            self.station_bikes[station_id] -= served
            served_total += served
            unmet_total += requested - served
        return served_total, unmet_total

    def _apply_returns(self, returns: list[int]) -> tuple[int, int]:
        """반납을 재고에 더하고 수용/거절 반납량을 반환한다."""
        accepted_total = 0
        rejected_total = 0
        for station_id, returned in enumerate(returns):
            space = self.config.station_capacity - self.station_bikes[station_id]
            accepted = min(space, returned)
            self.station_bikes[station_id] += accepted
            accepted_total += accepted
            rejected_total += returned - accepted
        return accepted_total, rejected_total

    def _info(
        self,
        reward: float,
        unmet: int,
        movement_cost: float,
        demand: list[int] | None = None,
        returns: list[int] | None = None,
        served: int = 0,
        accepted_returns: int = 0,
        rejected_returns: int = 0,
        moved_bikes: int = 0,
        traffic_factor: float = 1.0,
    ) -> dict[str, object]:
        """사람이 결과를 해석하기 위한 보조 정보를 만든다."""
        return {
            "time_step": self.time_step,
            "active_date": self.active_date,
            "station_names": list(self.config.station_names),
            # 사람이 평가 결과를 읽을 때 필요한 원본 상태와 step 결과를 담는다.
            "station_bikes": self.station_bikes.copy(),
            "truck_location": self.truck_location,
            "truck_bikes": self.truck_bikes,
            "demand": demand or [0 for _ in range(self.config.station_count)],
            "returns": returns or [0 for _ in range(self.config.station_count)],
            "served_demand": served,
            "unmet_demand": unmet,
            "accepted_returns": accepted_returns,
            "rejected_returns": rejected_returns,
            "movement_cost": movement_cost,
            "moved_bikes": moved_bikes,
            "traffic_factor": traffic_factor,
            "reward": reward,
        }


def _validate_traffic_factors(
    traffic_factors: tuple[float, ...],
    episode_steps: int,
) -> None:
    """traffic factor가 episode 시간 길이와 양수 조건을 만족하는지 검증한다."""
    if len(traffic_factors) != episode_steps:
        raise ValueError("traffic_factors length must match episode_steps")
    if any(factor <= 0 for factor in traffic_factors):
        raise ValueError("traffic_factors must be positive")


def _validate_hourly_ranges(
    name: str,
    ranges: dict[int, tuple[tuple[int, int], ...]],
    episode_steps: int,
    station_count: int,
) -> None:
    """시간대별 샘플링 범위가 episode 길이와 대여소 수에 맞는지 검증한다."""
    expected_hours = set(range(episode_steps))
    actual_hours = set(ranges)
    if actual_hours != expected_hours:
        missing = sorted(expected_hours - actual_hours)
        extra = sorted(actual_hours - expected_hours)
        raise ValueError(f"{name} hour keys mismatch. missing={missing}, extra={extra}")

    for hour, station_ranges in ranges.items():
        if len(station_ranges) != station_count:
            raise ValueError(
                f"{name}[{hour}] must have {station_count} station ranges"
            )
        for station_id, (low, high) in enumerate(station_ranges):
            if low < 0 or high < low:
                raise ValueError(
                    f"{name}[{hour}][{station_id}] must satisfy 0 <= low <= high"
                )


def _validate_daily_counts(
    dates: tuple[str, ...],
    demand_counts: tuple[tuple[tuple[int, ...], ...], ...],
    return_counts: tuple[tuple[tuple[int, ...], ...], ...],
    episode_steps: int,
    station_count: int,
) -> None:
    """daily profile count가 날짜/시간/대여소 구조를 만족하는지 검증한다."""
    if not dates and not demand_counts and not return_counts:
        return
    if not (len(dates) == len(demand_counts) == len(return_counts)):
        raise ValueError("daily profile dates, demand_counts, return_counts length mismatch")

    for name, counts in (
        ("daily_demand_counts", demand_counts),
        ("daily_return_counts", return_counts),
    ):
        for date_index, day_counts in enumerate(counts):
            if len(day_counts) != episode_steps:
                raise ValueError(f"{name}[{date_index}] must have {episode_steps} hours")
            for hour, station_counts in enumerate(day_counts):
                if len(station_counts) != station_count:
                    raise ValueError(
                        f"{name}[{date_index}][{hour}] must have {station_count} station counts"
                    )
                if any(value < 0 for value in station_counts):
                    raise ValueError(f"{name}[{date_index}][{hour}] must be non-negative")
