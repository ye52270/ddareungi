"""V0 평가에 사용할 baseline policy 모음."""

from __future__ import annotations

import random
from typing import Protocol

from ddareungi_rl.envs import ToyDdareungiEnv


class Policy(Protocol):
    """평가 코드가 사용하는 action 선택 인터페이스를 정의한다."""

    def select_action(self, env: ToyDdareungiEnv) -> int:
        """다음에 방문할 대여소 id를 선택한다."""


class RandomPolicy:
    """대여소를 균등 무작위로 선택하는 policy다."""

    def __init__(self, seed: int | None = None) -> None:
        """선택적 seed를 받아 random baseline policy를 만든다."""
        self.rng = random.Random(seed)

    def select_action(self, env: ToyDdareungiEnv) -> int:
        """환경의 action space에서 무작위 대여소 action을 선택한다."""
        return self.rng.randrange(env.action_space_n)


class LowStockPolicy:
    """자전거 수가 가장 적은 대여소를 방문하는 policy다."""

    def select_action(self, env: ToyDdareungiEnv) -> int:
        """현재 자전거 재고가 가장 낮은 대여소를 선택한다."""
        lowest_stock_station = 0
        for station_id in range(1, env.action_space_n):
            if env.station_bikes[station_id] < env.station_bikes[lowest_stock_station]:
                lowest_stock_station = station_id
        return lowest_stock_station


class DemandAwarePolicy:
    """현재 시간대의 예상 수요 부족분이 가장 큰 대여소를 방문하는 policy다."""

    def select_action(self, env: ToyDdareungiEnv) -> int:
        """예상 수요와 현재 재고 차이가 가장 큰 대여소를 선택한다."""
        expected_demand = self._expected_demand(env)
        best_station = 0
        best_score = self._shortage_score(env, expected_demand, 0)
        for station_id in range(1, env.action_space_n):
            score = self._shortage_score(env, expected_demand, station_id)
            if score > best_score:
                best_station = station_id
                best_score = score
        return best_station

    def _expected_demand(self, env: ToyDdareungiEnv) -> list[float]:
        """환경의 현재 시간대 demand pattern에서 대여소별 기대 수요를 계산한다."""
        hour = env.time_step % env.config.episode_steps
        for time_range, station_ranges in env.config.demand_pattern.items():
            if hour in time_range:
                return [(low + high) / 2 for low, high in station_ranges]
        return [0.0 for _ in range(env.action_space_n)]

    def _shortage_score(
        self,
        env: ToyDdareungiEnv,
        expected_demand: list[float],
        station_id: int,
    ) -> tuple[float, int]:
        """예상 수요 부족분과 낮은 재고를 함께 반영한 정렬 점수를 반환한다."""
        shortage = expected_demand[station_id] - env.station_bikes[station_id]
        return (shortage, -env.station_bikes[station_id])
