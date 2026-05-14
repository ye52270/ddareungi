"""V0 따릉이 재배치 toy 환경.

첫 MDP를 작게 유지하기 위해 다음 요소만 포함한다.

- 가상 대여소 3개
- 재배치 트럭 1대
- action = 다음 방문 대여소 선택
- 싣기/내리기 = target stock 기반 rule-based heuristic
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Literal

from ddareungi_rl.stations import STATION_NAMES


RenderMode = Literal["ansi", "human"]


def default_demand_pattern() -> dict[range, tuple[tuple[int, int], ...]]:
    """시간대별 toy 대여 수요 샘플링 범위를 반환한다."""
    return {
        range(0, 6): ((0, 1), (0, 1), (0, 1)),
        range(6, 11): ((3, 5), (1, 2), (1, 3)),
        range(11, 17): ((1, 2), (2, 3), (2, 4)),
        range(17, 22): ((1, 3), (3, 5), (1, 2)),
        range(22, 24): ((0, 1), (1, 2), (0, 1)),
    }


def default_return_pattern() -> dict[range, tuple[tuple[int, int], ...]]:
    """시간대별 toy 반납 수량 샘플링 범위를 반환한다."""
    return {
        range(0, 6): ((0, 1), (0, 1), (0, 1)),
        range(6, 11): ((0, 1), (2, 4), (0, 1)),
        range(11, 17): ((1, 2), (1, 2), (1, 3)),
        range(17, 22): ((2, 4), (0, 1), (0, 1)),
        range(22, 24): ((1, 2), (0, 1), (0, 1)),
    }


@dataclass(frozen=True)
class ToyDdareungiConfig:
    """Toy 따릉이 환경에서 조정 가능한 상수를 저장한다."""

    station_count: int = 3
    station_capacity: int = 10
    truck_capacity: int = 5
    target_stock: int = 5
    episode_steps: int = 24
    unmet_demand_penalty: int = 10
    movement_cost_value: int = 1
    initial_stock_min: int = 2
    initial_stock_max: int = 8
    initial_truck_bikes: int = 3
    initial_truck_location: int = 0
    demand_pattern: dict[range, tuple[tuple[int, int], ...]] = field(
        default_factory=default_demand_pattern
    )
    return_pattern: dict[range, tuple[tuple[int, int], ...]] = field(
        default_factory=default_return_pattern
    )


class ToyDdareungiEnv:
    """V0 실험을 위한 작은 Gymnasium 스타일 환경이다."""

    metadata = {"render_modes": [None, "ansi", "human"]}

    def __init__(
        self,
        config: ToyDdareungiConfig | None = None,
        render_mode: RenderMode | None = None,
        seed: int | None = None,
    ) -> None:
        """설정, 렌더링 모드, seed를 받아 환경 인스턴스를 만든다."""
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be one of None, 'ansi', or 'human'")

        self.config = config or ToyDdareungiConfig()
        self.render_mode = render_mode
        self.rng = random.Random(seed)
        self.station_bikes = [0] * self.config.station_count
        self.truck_location = self.config.initial_truck_location
        self.truck_bikes = self.config.initial_truck_bikes
        self.time_step = 0
        self.last_info: dict[str, object] = {}

    @property
    def action_space_n(self) -> int:
        """선택 가능한 대여소 action 개수를 반환한다."""
        return self.config.station_count

    @property
    def observation_size(self) -> int:
        """정규화된 observation vector의 길이를 반환한다."""
        return self.config.station_count + 3

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, object]]:
        """새 episode를 시작하고 초기 observation과 info를 반환한다."""
        if seed is not None:
            self.rng.seed(seed)

        self.station_bikes = [
            self.rng.randint(self.config.initial_stock_min, self.config.initial_stock_max)
            for _ in range(self.config.station_count)
        ]
        self.truck_location = self.config.initial_truck_location
        self.truck_bikes = self.config.initial_truck_bikes
        self.time_step = 0
        self.last_info = {
            "station_names": STATION_NAMES.copy(),
            "station_bikes": self.station_bikes.copy(),
            "truck_location": self.truck_location,
            "truck_bikes": self.truck_bikes,
            "time_step": self.time_step,
        }

        return self._observation(), self.last_info.copy()

    def step(
        self, action: int
    ) -> tuple[list[float], float, bool, bool, dict[str, object]]:
        """대여소 방문 action 하나를 적용하고 다음 transition tuple을 반환한다."""
        if not 0 <= action < self.config.station_count:
            raise ValueError(f"action must be between 0 and {self.config.station_count - 1}")

        previous_location = self.truck_location
        previous_station_bikes = self.station_bikes.copy()
        previous_truck_bikes = self.truck_bikes
        movement_cost = (
            self.config.movement_cost_value if action != previous_location else 0
        )

        self.truck_location = action
        station_before_rebalance = self.station_bikes[action]
        truck_before_rebalance = self.truck_bikes
        relocation_delta = self._auto_rebalance(action)
        station_after_rebalance = self.station_bikes[action]
        truck_after_rebalance = self.truck_bikes
        rebalance_type = self._rebalance_type(relocation_delta)
        rebalance_amount = abs(relocation_delta)
        demand = self._sample_demand(self.time_step)
        served_demand, unmet_demand = self._apply_demand(demand)
        returns = self._sample_returns(self.time_step)
        accepted_returns, full_returns = self._apply_returns(returns)

        service_success = unmet_demand == 0
        reward = -self.config.unmet_demand_penalty * unmet_demand - movement_cost

        self.time_step += 1
        terminated = False
        truncated = self.time_step >= self.config.episode_steps

        info = {
            "station_names": STATION_NAMES.copy(),
            "time_step": self.time_step,
            "previous_truck_location": previous_location,
            "truck_previous_location": previous_location,
            "truck_location": self.truck_location,
            "truck_bikes": self.truck_bikes,
            "station_bikes": self.station_bikes.copy(),
            "previous_station_bikes": previous_station_bikes,
            "previous_truck_bikes": previous_truck_bikes,
            "action": action,
            "demand": demand,
            "returns": returns,
            "served_demand": served_demand,
            "unmet_demand": unmet_demand,
            "accepted_returns": accepted_returns,
            "full_returns": full_returns,
            "movement_cost": movement_cost,
            "service_success": service_success,
            "relocation_delta": relocation_delta,
            "rebalance_type": rebalance_type,
            "rebalance_station": action,
            "rebalance_amount": rebalance_amount,
            "truck_event": rebalance_type if rebalance_amount else "move",
            "truck_event_amount": rebalance_amount,
            "station_bikes_before_rebalance": station_before_rebalance,
            "station_bikes_after_rebalance": station_after_rebalance,
            "truck_bikes_before_rebalance": truck_before_rebalance,
            "truck_bikes_after_rebalance": truck_after_rebalance,
            "reward": reward,
        }
        self.last_info = info

        if self.render_mode == "human":
            print(self.render())

        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        """현재 환경 상태를 text frame으로 렌더링한다."""
        frame = self._render_text()
        if self.render_mode == "human":
            return frame
        if self.render_mode == "ansi":
            return frame
        return frame

    def close(self) -> None:
        """렌더링 자원을 해제한다. 현재 text 렌더링에서는 no-op이다."""
        return None

    def _observation(self) -> list[float]:
        """원본 환경 상태에서 정규화된 observation vector를 만든다."""
        cfg = self.config
        normalized_stations = [bikes / cfg.station_capacity for bikes in self.station_bikes]
        return [
            *normalized_stations,
            self.truck_location / max(1, cfg.station_count - 1),
            self.truck_bikes / cfg.truck_capacity,
            self.time_step / cfg.episode_steps,
        ]

    def _auto_rebalance(self, station_id: int) -> int:
        """방문한 대여소의 재고가 target stock에 가까워지도록 싣거나 내린다."""
        current_stock = self.station_bikes[station_id]
        target = self.config.target_stock

        if current_stock > target:
            available_truck_space = self.config.truck_capacity - self.truck_bikes
            load_amount = min(current_stock - target, available_truck_space)
            self.station_bikes[station_id] -= load_amount
            self.truck_bikes += load_amount
            return -load_amount

        if current_stock < target:
            unload_amount = min(target - current_stock, self.truck_bikes)
            self.station_bikes[station_id] += unload_amount
            self.truck_bikes -= unload_amount
            return unload_amount

        return 0

    def _rebalance_type(self, relocation_delta: int) -> str:
        """재배치 변화량을 load/unload/none 문자열로 변환한다."""
        if relocation_delta < 0:
            return "load"
        if relocation_delta > 0:
            return "unload"
        return "none"

    def _sample_demand(self, time_step: int) -> list[int]:
        """현재 time step의 대여소별 대여 수요를 샘플링한다."""
        return self._sample_pattern(self.config.demand_pattern, time_step, "demand")

    def _sample_returns(self, time_step: int) -> list[int]:
        """현재 time step의 대여소별 반납 수량을 샘플링한다."""
        return self._sample_pattern(self.config.return_pattern, time_step, "return")

    def _sample_pattern(
        self,
        pattern: dict[range, tuple[tuple[int, int], ...]],
        time_step: int,
        pattern_name: str,
    ) -> list[int]:
        """시간대별 범위 pattern에서 대여소별 값을 하나씩 샘플링한다."""
        hour = time_step % self.config.episode_steps
        for time_range, station_ranges in pattern.items():
            if hour in time_range:
                return [self.rng.randint(low, high) for low, high in station_ranges]
        raise RuntimeError(f"No {pattern_name} pattern configured for hour {hour}")

    def _apply_demand(self, demand: list[int]) -> tuple[int, int]:
        """대여 가능한 수요를 재고에서 차감하고 미충족 수요를 계산한다."""
        served_total = 0
        unmet_total = 0

        for station_id, requested in enumerate(demand):
            served = min(self.station_bikes[station_id], requested)
            unmet = requested - served
            self.station_bikes[station_id] -= served
            served_total += served
            unmet_total += unmet

        return served_total, unmet_total

    def _apply_returns(self, returns: list[int]) -> tuple[int, int]:
        """반납 자전거를 재고에 더하고 capacity 때문에 거절된 반납을 계산한다."""
        accepted_total = 0
        full_total = 0

        for station_id, returned in enumerate(returns):
            available_space = self.config.station_capacity - self.station_bikes[station_id]
            accepted = min(available_space, returned)
            rejected = returned - accepted
            self.station_bikes[station_id] += accepted
            accepted_total += accepted
            full_total += rejected

        return accepted_total, full_total

    def _render_text(self) -> str:
        """현재 상태와 마지막 transition 정보를 ANSI 스타일 frame으로 포맷한다."""
        cells = []
        for station_id, bikes in enumerate(self.station_bikes):
            label = STATION_NAMES[station_id] if station_id < len(STATION_NAMES) else f"S{station_id}"
            truck_marker = " T" if self.truck_location == station_id else "  "
            cells.append(f"{label}{truck_marker} bikes={bikes:02d}")

        last_reward = self.last_info.get("reward", "-")
        last_unmet = self.last_info.get("unmet_demand", "-")
        last_demand = self.last_info.get("demand", "-")
        last_returns = self.last_info.get("returns", "-")

        return "\n".join(
            [
                "+---------------- Toy Ddareungi V0 ----------------+",
                f"| time={self.time_step:02d}/{self.config.episode_steps} "
                f"truck_loc={self.truck_location} truck_bikes={self.truck_bikes} |",
                f"| {cells[0]:<18} | {cells[1]:<18} |",
                f"| {cells[2]:<18} | demand={str(last_demand):<15} |",
                f"| returns={str(last_returns):<15} | reward={str(last_reward):<6} unmet={str(last_unmet):<6} |",
                "+--------------------------------------------------+",
            ]
        )
