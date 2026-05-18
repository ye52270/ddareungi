"""기본 DQN 알고리즘.

핵심 수식:
    target = r + gamma * max_a' Q_target(s', a')
    loss = (target - Q_online(s, a))^2

online network가 현재 Q(s, a)를 예측하고, target network가 다음 상태의
최대 Q값을 평가한다.
"""

from __future__ import annotations

import torch
from torch import nn

from ddareungi_rl.algorithms.common import DQNConfig, GreedyQPolicy, train_q_learning
from ddareungi_rl.algorithms.networks import QNetwork
from ddareungi_rl.env import DdareungiEnv


def train_dqn(
    env: DdareungiEnv,
    config: DQNConfig,
    seed: int = 42,
    verbose: bool = False,
    log_interval: int = 50,
) -> tuple[GreedyQPolicy, list[dict[str, float]]]:
    """기본 QNetwork와 DQN target으로 학습한다."""
    return train_q_learning(
        env=env,
        config=config,
        network_cls=QNetwork,
        target_fn=compute_dqn_target,
        seed=seed,
        verbose=verbose,
        log_interval=log_interval,
        log_label="dqn-train",
    )


def compute_dqn_target(
    online: nn.Module,
    target: nn.Module,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    done: torch.Tensor,
    config: DQNConfig,
) -> torch.Tensor:
    """DQN TD target을 계산한다: r + gamma * max_a' Q_target(s', a')."""
    del online
    next_q_values = target(next_state).max(dim=1).values
    return reward + config.gamma * next_q_values * (1.0 - done)

