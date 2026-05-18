"""따릉이 재배치를 위한 최소 Gymnasium 환경."""

from __future__ import annotations

from dataclasses import dataclass, field
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def default_station_names() -> tuple[str, str, str]:
    """기본 toy 대여소 이름을 반환한다."""
    return ("마곡나루역", "LG사옥", "마곡수명산")


def default_demand_ranges() -> dict[int, tuple[tuple[int, int], ...]]:
    """시간대별 대여 수요 샘플링 범위를 반환한다."""
    return {
        0: ((0, 1), (0, 1), (0, 1)),
        1: ((0, 1), (0, 1), (0, 1)),
        2: ((0, 1), (0, 1), (0, 1)),
        3: ((0, 1), (0, 1), (0, 1)),
        4: ((0, 1), (0, 1), (0, 1)),
        5: ((0, 1), (0, 1), (0, 1)),
        6: ((2, 4), (0, 1), (1, 2)),
        7: ((3, 5), (1, 2), (2, 4)),
        8: ((3, 5), (1, 2), (2, 4)),
        9: ((2, 4), (1, 2), (1, 3)),
        10: ((1, 3), (1, 2), (1, 2)),
        11: ((1, 2), (1, 3), (1, 2)),
        12: ((1, 2), (1, 3), (1, 2)),
        13: ((1, 2), (1, 3), (1, 2)),
        14: ((1, 2), (1, 3), (1, 2)),
        15: ((1, 2), (2, 3), (1, 2)),
        16: ((1, 3), (3, 4), (1, 2)),
        17: ((2, 4), (3, 5), (1, 2)),
        18: ((2, 4), (3, 5), (1, 2)),
        19: ((1, 3), (2, 4), (1, 2)),
        20: ((1, 2), (1, 3), (1, 2)),
        21: ((1, 2), (1, 2), (1, 2)),
        22: ((0, 1), (1, 2), (0, 1)),
        23: ((0, 1), (0, 1), (0, 1)),
    }


def default_return_ranges() -> dict[int, tuple[tuple[int, int], ...]]:
    """시간대별 반납 수량 샘플링 범위를 반환한다."""
    return {
        hour: ((0, 1), (0, 1), (0, 1)) for hour in range(24)
    } | {
        7: ((1, 2), (2, 4), (0, 1)),
        8: ((1, 2), (3, 5), (0, 1)),
        17: ((2, 4), (1, 2), (0, 1)),
        18: ((2, 4), (1, 2), (0, 1)),
        19: ((1, 3), (1, 2), (1, 2)),
    }


@dataclass(frozen=True)
class EnvConfig:
    """환경의 크기, 보상, 수요 패턴을 보관한다."""

    station_names: tuple[str, ...] = field(default_factory=default_station_names)
    station_capacity: int = 10
    truck_capacity: int = 5
    target_stock: int = 5
    episode_steps: int = 24
    unmet_penalty: int = 10
    full_penalty: int = 3
    move_cost: int = 1
    initial_truck_bikes: int = 3
    demand_ranges: dict[int, tuple[tuple[int, int], ...]] = field(
        default_factory=default_demand_ranges
    )
    return_ranges: dict[int, tuple[tuple[int, int], ...]] = field(
        default_factory=default_return_ranges
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
        self.config = config or EnvConfig()
        self.rng = random.Random(seed)
        self.action_space = spaces.Discrete(self.config.station_count)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.config.station_count + 3,),
            dtype=np.float32,
        )
        self.station_bikes = [0 for _ in range(self.config.station_count)]
        self.truck_location = 0
        self.truck_bikes = self.config.initial_truck_bikes
        self.time_step = 0
        self.last_info: dict[str, object] = {}

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        """새 episode를 시작하고 observation과 info를 반환한다."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            self.action_space.seed(seed)
        self.station_bikes = [
            self.rng.randint(2, 8) for _ in range(self.config.station_count)
        ]
        self.truck_location = 0
        self.truck_bikes = self.config.initial_truck_bikes
        self.time_step = 0
        self.last_info = self._info(reward=0.0, unmet=0, movement_cost=0)
        return self._observation(), self.last_info.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """방문할 대여소 action을 적용하고 다음 상태와 보상을 반환한다."""
        action = int(action)
        if action < 0 or action >= self.config.station_count:
            raise ValueError(f"action must be between 0 and {self.config.station_count - 1}")

        movement_cost = self.config.move_cost if action != self.truck_location else 0
        self.truck_location = action
        moved_bikes = self._rebalance(action)
        demand = self._sample(self.config.demand_ranges)
        served, unmet = self._apply_demand(demand)
        returns = self._sample(self.config.return_ranges)
        accepted_returns, rejected_returns = self._apply_returns(returns)
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
        )
        self.last_info = info
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
        values = [
            *(bike / cfg.station_capacity for bike in self.station_bikes),
            self.truck_location / max(1, cfg.station_count - 1),
            self.truck_bikes / cfg.truck_capacity,
            self.time_step / cfg.episode_steps,
        ]
        return np.array(values, dtype=np.float32)

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
        movement_cost: int,
        demand: list[int] | None = None,
        returns: list[int] | None = None,
        served: int = 0,
        accepted_returns: int = 0,
        rejected_returns: int = 0,
        moved_bikes: int = 0,
    ) -> dict[str, object]:
        """사람이 결과를 해석하기 위한 보조 정보를 만든다."""
        return {
            "time_step": self.time_step,
            "station_names": list(self.config.station_names),
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
            "reward_formula": "-unmet_penalty * unmet_demand - full_penalty * rejected_returns - movement_cost",
            "reward": reward,
        }
