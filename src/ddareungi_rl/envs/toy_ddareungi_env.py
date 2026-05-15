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
    """시간대별 toy 대여 수요 샘플링 범위를 반환한다.

    각 tuple은 대여소별 ``(최소, 최대)`` 범위다.
    예: 6~10시에는 0번 대여소에서 3~5대 수요가 발생할 수 있다.
    """
    return {
        range(0, 6): ((0, 1), (0, 1), (0, 1)),
        range(6, 11): ((3, 5), (1, 2), (1, 3)),
        range(11, 17): ((1, 2), (2, 3), (2, 4)),
        range(17, 22): ((1, 3), (3, 5), (1, 2)),
        range(22, 24): ((0, 1), (1, 2), (0, 1)),
    }


def default_return_pattern() -> dict[range, tuple[tuple[int, int], ...]]:
    """시간대별 toy 반납 수량 샘플링 범위를 반환한다.

    demand pattern과 같은 구조이며, 각 대여소에 들어오는 반납량을 샘플링한다.
    예: 6~10시에는 1번 대여소에서 2~4대 반납이 발생할 수 있다.
    """
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

    # 대여소/트럭의 물리적인 크기를 정한다.
    station_count: int = 3
    station_capacity: int = 10
    truck_capacity: int = 5

    # 트럭이 방문한 대여소를 몇 대 수준으로 맞출지 정하는 rule-based 기준이다.
    target_stock: int = 5

    # 24 step을 하루 24시간처럼 해석하는 toy episode 길이다.
    episode_steps: int = 24

    # reward 설계의 핵심 계수: 헛걸음은 크게, 이동은 작게 벌점화한다.
    unmet_demand_penalty: int = 10
    movement_cost_value: int = 1

    # reset() 때 대여소별 초기 재고와 트럭 초기 상태를 만든다.
    initial_stock_min: int = 2
    initial_stock_max: int = 8
    initial_truck_bikes: int = 3
    initial_truck_location: int = 0

    # mutable dict를 모든 인스턴스가 공유하지 않도록 default_factory를 사용한다.
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

        # config가 없으면 기본 toy MDP 설정으로 시작한다.
        self.config = config or ToyDdareungiConfig()
        self.render_mode = render_mode

        # 환경 내부 난수만 따로 관리하면 같은 seed로 같은 episode를 재현할 수 있다.
        self.rng = random.Random(seed)

        # 아래 값들이 observation의 원본 상태다. reset()에서 episode 초기값으로 다시 설정된다.
        self.station_bikes = [0] * self.config.station_count
        self.truck_location = self.config.initial_truck_location
        self.truck_bikes = self.config.initial_truck_bikes
        self.time_step = 0

        # info는 학습 입력이 아니라 로그/시각화/디버깅을 위한 설명 데이터다.
        self.last_info: dict[str, object] = {}

    @property
    def action_space_n(self) -> int:
        """선택 가능한 대여소 action 개수를 반환한다."""
        return self.config.station_count

    @property
    def observation_size(self) -> int:
        """정규화된 observation vector의 길이를 반환한다."""
        return self.config.station_count + 3

    def current_observation(self) -> list[float]:
        """현재 환경 상태의 정규화된 observation vector를 반환한다."""
        return self._observation()

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, object]]:
        """새 episode를 시작하고 초기 observation과 info를 반환한다."""
        if seed is not None:
            self.rng.seed(seed)

        # episode마다 대여소 초기 재고를 조금씩 바꿔서 agent가 한 상황에만 외우지 않게 한다.
        self.station_bikes = [
            self.rng.randint(self.config.initial_stock_min, self.config.initial_stock_max)
            for _ in range(self.config.station_count)
        ]

        # 트럭과 시간은 항상 같은 기준점에서 하루를 시작한다.
        self.truck_location = self.config.initial_truck_location
        self.truck_bikes = self.config.initial_truck_bikes
        self.time_step = 0

        # reset 직후에도 GUI/로그가 현재 상태를 바로 설명할 수 있게 info를 채운다.
        self.last_info = {
            "station_names": STATION_NAMES.copy(),
            "station_bikes": self.station_bikes.copy(),
            "truck_location": self.truck_location,
            "truck_bikes": self.truck_bikes,
            "time_step": self.time_step,
        }
        # _observation: 에이전트가 학습에 사용할 숫자 vector
        # last_info: 사람이 해석하기 좋은 부가 정보
        return self._observation(), self.last_info.copy()

    def step(
        self, action: int
    ) -> tuple[list[float], float, bool, bool, dict[str, object]]:
        """대여소 방문 action 하나를 적용하고 다음 transition tuple을 반환한다.

        이 함수가 MDP의 transition이다. agent는 방문할 대여소만 고르고,
        환경은 이동, 자동 재배치, 수요/반납, reward 계산을 순서대로 처리한다.
        """
        if not 0 <= action < self.config.station_count:
            raise ValueError(f"action must be between 0 and {self.config.station_count - 1}")

        # 정책이 action을 고른 '선택 전 상태'를 남긴다. GUI에서는 이 값으로 판단 근거를 설명한다.
        previous_location = self.truck_location
        previous_station_bikes = self.station_bikes.copy()
        previous_truck_bikes = self.truck_bikes

        # 같은 대여소를 다시 선택하면 트럭이 이동하지 않은 것으로 보고 이동비용을 0으로 둔다.
        movement_cost = (
            self.config.movement_cost_value if action != previous_location else 0
        )

        # action은 "트럭이 다음에 방문할 대여소 id"다.
        self.truck_location = action
        station_before_rebalance = self.station_bikes[action]
        truck_before_rebalance = self.truck_bikes

        # agent는 방문지만 고르고, 싣기/내리기는 target_stock 기준 환경 규칙이 처리한다.
        relocation_delta = self._auto_rebalance(action)

        station_after_rebalance = self.station_bikes[action]
        truck_after_rebalance = self.truck_bikes
        station_bikes_after_rebalance = self.station_bikes.copy()
        rebalance_type = self._rebalance_type(relocation_delta)
        rebalance_amount = abs(relocation_delta)

        # 이 toy MDP에서는 재배치가 먼저 일어나고, 그 뒤 한 시간 동안의 대여 수요가 발생한다.
        demand = self._sample_demand(self.time_step)
        served_demand, unmet_demand = self._apply_demand(demand)
        station_bikes_after_demand = self.station_bikes.copy()

        # 수요 처리 후 반납이 들어온다고 가정한다. capacity 초과분은 full_returns로 기록된다.
        returns = self._sample_returns(self.time_step)
        accepted_returns, full_returns = self._apply_returns(returns)

        service_success = unmet_demand == 0

        # reward는 "헛걸음 감소"를 가장 크게 보고, 불필요한 이동은 작은 벌점으로 반영한다.
        reward = -self.config.unmet_demand_penalty * unmet_demand - movement_cost

        self.time_step += 1

        # 현재 V0에는 실패로 즉시 종료되는 terminal state가 없고, 하루가 끝나면 truncated가 된다.
        terminated = False
        truncated = self.time_step >= self.config.episode_steps

        # info는 학습에는 직접 쓰지 않고, replay/평가/시각화에서 transition을 설명하기 위해 남긴다.
        info = {
            "station_names": STATION_NAMES.copy(),
            "time_step": self.time_step,
            "previous_truck_location": previous_location,
            "truck_previous_location": previous_location,
            "truck_location": self.truck_location,
            "truck_bikes": self.truck_bikes,
            "station_bikes": self.station_bikes.copy(),
            "decision_station_bikes": previous_station_bikes,
            "decision_truck_location": previous_location,
            "decision_truck_bikes": previous_truck_bikes,
            "same_location_action": action == previous_location,
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
            "station_bikes_after_rebalance_all": station_bikes_after_rebalance,
            "station_bikes_after_demand": station_bikes_after_demand,
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

        # DQN 같은 신경망이 다루기 쉽도록 재고/위치/적재량/시간을 0~1 범위로 맞춘다.
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

        # 대여소에 자전거가 너무 많으면 트럭에 싣는다. 반환값 음수는 load를 뜻한다.
        if current_stock > target:
            available_truck_space = self.config.truck_capacity - self.truck_bikes
            load_amount = min(current_stock - target, available_truck_space)
            self.station_bikes[station_id] -= load_amount
            self.truck_bikes += load_amount
            return -load_amount

        # 대여소가 부족하면 트럭에서 내린다. 반환값 양수는 unload를 뜻한다.
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
                # Python range(6, 11)은 6~10시를 의미한다. high는 randint 특성상 포함된다.
                return [self.rng.randint(low, high) for low, high in station_ranges]
        raise RuntimeError(f"No {pattern_name} pattern configured for hour {hour}")

    def _apply_demand(self, demand: list[int]) -> tuple[int, int]:
        """대여 가능한 수요를 재고에서 차감하고 미충족 수요를 계산한다."""
        served_total = 0
        unmet_total = 0

        for station_id, requested in enumerate(demand):
            # 재고가 부족하면 처리하지 못한 요청이 unmet demand, 즉 헛걸음으로 남는다.
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
            # 대여소가 꽉 차 있으면 일부 반납은 받아주지 못하고 full_returns로 기록된다.
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
