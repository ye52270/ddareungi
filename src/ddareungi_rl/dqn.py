"""교육용으로 단순화한 PyTorch DQN 학습 코드."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Protocol

import numpy as np
import torch
from torch import nn

from ddareungi_rl.env import DdareungiEnv


@dataclass
class DQNConfig:
    """DQN 학습에 필요한 하이퍼파라미터를 보관한다."""

    episodes: int = 100
    gamma: float = 0.95
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 1000
    replay_size: int = 3000
    batch_size: int = 32
    min_replay: int = 100
    target_update: int = 100
    hidden_size: int = 64


class Policy(Protocol):
    """평가 함수가 요구하는 policy 인터페이스."""

    def act(self, env: DdareungiEnv) -> int:
        """현재 환경에서 action을 선택한다."""


class QNetwork(nn.Module):
    """작은 MLP로 Q(s, a)를 근사한다."""

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


class DQNPolicy:
    """학습된 Q-network로 greedy action을 선택한다."""

    def __init__(self, network: QNetwork) -> None:
        """Q-network를 받아 평가용 policy를 만든다."""
        self.network = network

    def act(self, env: DdareungiEnv) -> int:
        """현재 observation에서 Q value가 가장 큰 action을 반환한다."""
        state = torch.tensor(env._observation(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.network(state), dim=1).item())


def epsilon_by_step(step: int, config: DQNConfig) -> float:
    """현재 step의 epsilon 값을 선형 감소 방식으로 계산한다."""
    ratio = min(1.0, step / config.epsilon_decay)
    return config.epsilon_start + ratio * (config.epsilon_end - config.epsilon_start)


def train_dqn(
    env: DdareungiEnv,
    config: DQNConfig,
    seed: int = 42,
) -> tuple[DQNPolicy, list[dict[str, float]]]:
    """환경에서 DQN을 학습하고 평가용 policy와 episode metrics를 반환한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    online = QNetwork(
        env.observation_space.shape[0],
        env.action_space.n,
        config.hidden_size,
    )
    target = QNetwork(
        env.observation_space.shape[0],
        env.action_space.n,
        config.hidden_size,
    )
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
    metrics: list[dict[str, float]] = []
    global_step = 0

    for episode in range(config.episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_unmet = 0

        while not done:
            epsilon = epsilon_by_step(global_step, config)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = DQNPolicy(online).act(env)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.append((state, action, reward, next_state, done))
            replay = replay[-config.replay_size :]
            episode_reward += reward
            episode_unmet += int(info["unmet_demand"])

            if len(replay) >= config.min_replay:
                _learn_from_batch(online, target, optimizer, replay, config)
            if global_step % config.target_update == 0:
                target.load_state_dict(online.state_dict())

            state = next_state
            global_step += 1

        metrics.append(
            {
                "episode": float(episode + 1),
                "reward": episode_reward,
                "unmet_demand": float(episode_unmet),
                "epsilon": epsilon_by_step(global_step, config),
            }
        )

    return DQNPolicy(online), metrics


def evaluate_policy(
    env: DdareungiEnv,
    policy: Policy,
    episodes: int = 5,
    seed: int = 1000,
) -> dict[str, float]:
    """policy를 여러 episode에서 평가하고 평균 지표를 반환한다."""
    rewards = []
    unmet_values = []
    rejected_return_values = []
    service_rates = []
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        done = False
        reward_sum = 0.0
        served_sum = 0
        unmet_sum = 0
        rejected_return_sum = 0
        while not done:
            _, reward, terminated, truncated, info = env.step(policy.act(env))
            done = terminated or truncated
            reward_sum += reward
            served_sum += int(info["served_demand"])
            unmet_sum += int(info["unmet_demand"])
            rejected_return_sum += int(info["rejected_returns"])
        total_demand = served_sum + unmet_sum
        rewards.append(reward_sum)
        unmet_values.append(unmet_sum)
        rejected_return_values.append(rejected_return_sum)
        service_rates.append(served_sum / total_demand if total_demand else 1.0)
    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_unmet_demand": float(np.mean(unmet_values)),
        "avg_rejected_returns": float(np.mean(rejected_return_values)),
        "avg_service_rate": float(np.mean(service_rates)),
    }


def save_model(policy: DQNPolicy, path: Path) -> None:
    """학습된 DQN policy의 network weight를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.network.state_dict(), path)


def _learn_from_batch(
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: list[tuple[np.ndarray, int, float, np.ndarray, bool]],
    config: DQNConfig,
) -> None:
    """replay buffer에서 batch를 뽑아 DQN 손실을 한 번 학습한다."""
    batch = random.sample(replay, config.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    action_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    next_state_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
    done_tensor = torch.tensor(dones, dtype=torch.float32)

    q_values = online(state_tensor).gather(1, action_tensor).squeeze(1)
    with torch.no_grad():
        next_q_values = target(next_state_tensor).max(dim=1).values
        targets = reward_tensor + config.gamma * next_q_values * (1.0 - done_tensor)

    loss = nn.functional.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
