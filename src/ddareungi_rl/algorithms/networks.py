"""DQN 계열 알고리즘에서 사용하는 Q-network 모음."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """기본 DQN의 Q(s, a)를 근사하는 작은 MLP."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        """입력 크기, action 수, hidden 크기로 네트워크를 만든다."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state batch를 받아 action별 Q value를 반환한다."""
        return self.layers(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN의 Q-network.

    수식:
        Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

    상태 자체의 가치 V(s)와 action별 상대 이점 A(s, a)를 분리해서
    action 차이가 작을 때도 상태 가치를 더 안정적으로 학습하게 한다.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        """공통 feature layer와 value/advantage head를 만든다."""
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_size, 1)
        self.advantage = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Dueling 수식으로 action별 Q value를 반환한다."""
        features = self.feature(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

