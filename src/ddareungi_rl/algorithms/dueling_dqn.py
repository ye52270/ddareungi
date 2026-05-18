"""Dueling DQN 알고리즘.

핵심 수식:
    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
    target = r + gamma * max_a' Q_target(s', a')

학습 target은 기본 DQN과 같고, Q-network 구조만 Dueling 구조를 사용한다.
"""

from __future__ import annotations

from ddareungi_rl.algorithms.common import DQNConfig, GreedyQPolicy, train_q_learning
from ddareungi_rl.algorithms.dqn import compute_dqn_target
from ddareungi_rl.algorithms.networks import DuelingQNetwork
from ddareungi_rl.env import DdareungiEnv


def train_dueling_dqn(
    env: DdareungiEnv,
    config: DQNConfig,
    seed: int = 42,
    verbose: bool = False,
    log_interval: int = 50,
) -> tuple[GreedyQPolicy, list[dict[str, float]]]:
    """DuelingQNetwork와 DQN target으로 학습한다."""
    return train_q_learning(
        env=env,
        config=config,
        network_cls=DuelingQNetwork,
        target_fn=compute_dqn_target,
        seed=seed,
        verbose=verbose,
        log_interval=log_interval,
        log_label="dueling-train",
    )

