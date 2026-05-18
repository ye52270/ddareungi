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

from ddareungi_rl.algorithms.common import (
    DQNConfig,
    GreedyQPolicy,
    Transition,
    choose_epsilon_greedy_action,
    epsilon_by_step,
    print_training_progress,
    sample_replay_batch,
    seed_everything,
    should_log_training,
)
from ddareungi_rl.algorithms.networks import QNetwork
from ddareungi_rl.env import DdareungiEnv


def train_dqn(
    env: DdareungiEnv,
    config: DQNConfig,
    seed: int = 42,
    verbose: bool = False,
    log_interval: int = 50,
) -> tuple[GreedyQPolicy, list[dict[str, float]]]:
    """
    기본 DQN을 episode 단위로 학습한다.

    흐름:
        1. online network와 target network를 만든다.
        2. epsilon-greedy로 action을 고른다.
        3. transition을 replay buffer에 저장한다.
        4. replay batch로 DQN TD target을 계산해 online network를 업데이트한다.
        5. 일정 step마다 target network를 online network로 동기화한다.
    """
    seed_everything(seed)
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
    policy = GreedyQPolicy(online)
    replay: list[Transition] = []
    metrics: list[dict[str, float]] = []
    global_step = 0

    for episode in range(config.episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_unmet = 0
        episode_rejected_returns = 0
        episode_movement_cost = 0.0
        episode_losses = []

        while not done:
            epsilon = epsilon_by_step(global_step, config)
            action = choose_epsilon_greedy_action(env, policy, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay.append((state, action, reward, next_state, done))
            replay = replay[-config.replay_size :]
            episode_reward += reward
            episode_unmet += int(info["unmet_demand"])
            episode_rejected_returns += int(info["rejected_returns"])
            episode_movement_cost += float(info["movement_cost"])

            if len(replay) >= config.min_replay:
                episode_losses.append(train_dqn_batch(online, target, optimizer, replay, config))
            if global_step % config.target_update == 0:
                target.load_state_dict(online.state_dict())

            state = next_state
            global_step += 1

        metrics.append(
            {
                "episode": float(episode + 1),
                "reward": episode_reward,
                "unmet_demand": float(episode_unmet),
                "rejected_returns": float(episode_rejected_returns),
                "movement_cost": float(episode_movement_cost),
                "epsilon": epsilon_by_step(global_step, config),
                "loss": _mean(episode_losses),
            }
        )
        if verbose and should_log_training(episode + 1, config.episodes, log_interval):
            print_training_progress(
                label="dqn-train",
                episode=episode + 1,
                total_episodes=config.episodes,
                metrics=metrics,
                log_interval=log_interval,
            )

    return policy, metrics


def train_dqn_batch(
    online: nn.Module,
    target: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: list[Transition],
    config: DQNConfig,
) -> float:
    """replay batch 하나로 DQN 손실을 계산하고 online network를 업데이트한다."""
    state, action, reward, next_state, done = sample_replay_batch(
        replay,
        config.batch_size,
    )
    predicted_q = online(state).gather(1, action).squeeze(1)
    with torch.no_grad():
        td_target = compute_dqn_target(online, target, reward, next_state, done, config)

    loss = nn.functional.mse_loss(predicted_q, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


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


def _mean(values: list[float]) -> float:
    """비어 있는 list는 0.0, 값이 있으면 평균을 반환한다."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))
