"""Double DQN 알고리즘.

핵심 수식:
    best_action = argmax_a' Q_online(s', a')
    target = r + gamma * Q_target(s', best_action)

DQN은 target network에서 바로 max를 취해 Q-value를 과대평가할 수 있다.
Double DQN은 online network가 action을 고르고, target network가 그 action의
값을 평가하게 하여 과대평가를 줄인다.
"""

from __future__ import annotations

import torch
from torch import nn

from ddareungi_rl.algorithms.common import DQNConfig, GreedyQPolicy, train_q_learning
from ddareungi_rl.algorithms.networks import QNetwork
from ddareungi_rl.env import DdareungiEnv


def train_double_dqn(
    env: DdareungiEnv,
    config: DQNConfig,
    seed: int = 42,
    verbose: bool = False,
    log_interval: int = 50,
) -> tuple[GreedyQPolicy, list[dict[str, float]]]:
    """기본 QNetwork와 Double DQN target으로 학습한다."""
    return train_q_learning(
        env=env,
        config=config,
        network_cls=QNetwork,
        target_fn=compute_double_dqn_target,
        seed=seed,
        verbose=verbose,
        log_interval=log_interval,
        log_label="double-dqn-train",
    )


def compute_double_dqn_target(
    online: nn.Module,
    target: nn.Module,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    done: torch.Tensor,
    config: DQNConfig,
) -> torch.Tensor:
    """Double DQN TD target을 계산한다."""
    best_next_actions = online(next_state).argmax(dim=1, keepdim=True)
    next_q_values = target(next_state).gather(1, best_next_actions).squeeze(1)
    return reward + config.gamma * next_q_values * (1.0 - done)

