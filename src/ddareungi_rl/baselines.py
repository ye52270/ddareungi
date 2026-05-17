"""DQN과 비교할 최소 baseline policy."""

from __future__ import annotations

import random

from ddareungi_rl.env import DdareungiEnv


class RandomPolicy:
    """무작위로 다음 방문 대여소를 선택한다."""

    def __init__(self, seed: int | None = None) -> None:
        """선택적 seed를 받아 random generator를 초기화한다."""
        self.rng = random.Random(seed)

    def act(self, env: DdareungiEnv) -> int:
        """환경 action space에서 무작위 action을 반환한다."""
        return self.rng.randrange(env.config.station_count)


class LowStockPolicy:
    """현재 자전거가 가장 적은 대여소를 선택한다."""

    def act(self, env: DdareungiEnv) -> int:
        """재고가 가장 낮은 대여소 index를 반환한다."""
        return min(range(env.config.station_count), key=lambda i: env.station_bikes[i])


class DemandAwarePolicy:
    """현재 시간대 예상 수요가 재고보다 큰 대여소를 우선 방문한다."""

    def act(self, env: DdareungiEnv) -> int:
        """예상 부족량이 가장 큰 대여소 index를 반환한다."""
        hour = env.time_step % env.config.episode_steps
        expected = [
            (low + high) / 2 for low, high in env.config.demand_ranges[hour]
        ]
        return max(
            range(env.config.station_count),
            key=lambda i: (expected[i] - env.station_bikes[i], -env.station_bikes[i]),
        )
