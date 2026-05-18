"""DQN 계열 알고리즘이 공유하는 설정, policy, 평가 helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from ddareungi_rl.env import DdareungiEnv


Transition = tuple[np.ndarray, int, float, np.ndarray, bool]
ReplayBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class Policy(Protocol):
    """평가 함수가 요구하는 policy 인터페이스."""

    def act(self, env: DdareungiEnv) -> int:
        """현재 환경에서 action을 선택한다."""


class GreedyQPolicy:
    """학습된 Q-network로 greedy action을 선택한다."""

    def __init__(self, network: nn.Module) -> None:
        """Q-network를 받아 평가용 policy를 만든다."""
        self.network = network

    def act(self, env: DdareungiEnv) -> int:
        """현재 observation에서 Q value가 가장 큰 action을 반환한다."""
        state = torch.tensor(env._observation(), dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            return int(torch.argmax(self.network(state), dim=1).item())


@dataclass
class DQNConfig:
    """DQN 계열 학습에 필요한 하이퍼파라미터를 보관한다."""

    episodes: int = 1000
    gamma: float = 0.95
    learning_rate: float = 0.0005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 8000
    replay_size: int = 10000
    batch_size: int = 32
    min_replay: int = 100
    target_update: int = 200
    hidden_size: int = 128


def epsilon_by_step(step: int, config: DQNConfig) -> float:
    """현재 step의 epsilon 값을 선형 감소 방식으로 계산한다."""
    ratio = min(1.0, step / config.epsilon_decay)
    return config.epsilon_start + ratio * (config.epsilon_end - config.epsilon_start)


def seed_everything(seed: int) -> None:
    """Python, NumPy, PyTorch 난수 seed를 한 번에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_replay_batch(replay: list[Transition], batch_size: int) -> ReplayBatch:
    """replay buffer에서 batch를 뽑아 PyTorch tensor 묶음으로 바꾼다."""
    batch = random.sample(replay, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    action_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    next_state_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
    done_tensor = torch.tensor(dones, dtype=torch.float32)
    return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor


def choose_epsilon_greedy_action(
    env: DdareungiEnv,
    policy: GreedyQPolicy,
    epsilon: float,
) -> int:
    """epsilon 확률로 무작위 action, 그 외에는 Q-network greedy action을 고른다."""
    if random.random() < epsilon:
        return int(env.action_space.sample())
    return policy.act(env)


def should_log_training(episode: int, total_episodes: int, log_interval: int) -> bool:
    """학습 진행 상황을 출력할 episode인지 판단한다."""
    return episode == 1 or episode == total_episodes or episode % log_interval == 0


def print_training_progress(
    *,
    label: str,
    episode: int,
    total_episodes: int,
    metrics: list[dict[str, float]],
    log_interval: int,
) -> None:
    """최근 episode 평균 핵심 지표를 한 줄 요약으로 콘솔에 출력한다."""
    recent_metrics = metrics[-log_interval:]
    window_size = len(recent_metrics)
    recent_reward = _average_metric(recent_metrics, "reward")
    recent_unmet = _average_metric(recent_metrics, "unmet_demand")
    recent_rejected = _average_metric(recent_metrics, "rejected_returns")
    recent_movement = _average_metric(recent_metrics, "movement_cost")
    recent_loss = _average_metric(recent_metrics, "loss")
    progress = _progress_bar(episode, total_episodes)
    print(
        f"[{label}] episode {episode:03d}/{total_episodes} {progress} "
        f"avg{window_size}_reward={recent_reward:.2f} "
        f"unmet={recent_unmet:.2f} "
        f"rejected={recent_rejected:.2f} "
        f"move={recent_movement:.2f} "
        f"loss={recent_loss:.4f} "
        f"eps={metrics[-1]['epsilon']:.3f}"
    )


def evaluate_policy(
    env: DdareungiEnv,
    policy: Policy,
    episodes: int = 5,
    seed: int = 1000,
    verbose: bool = False,
    label: str = "policy",
    sequential_dates: bool = True,
) -> dict[str, float]:
    """policy를 여러 episode에서 평가하고 평균 지표를 반환한다."""
    return evaluate_policy_with_trace(
        env=env,
        policy=policy,
        episodes=episodes,
        seed=seed,
        verbose=verbose,
        label=label,
        sequential_dates=sequential_dates,
    )["summary"]


def evaluate_policy_with_trace(
    env: DdareungiEnv,
    policy: Policy,
    episodes: int = 5,
    seed: int = 1000,
    verbose: bool = False,
    label: str = "policy",
    sequential_dates: bool = True,
) -> dict[str, Any]:
    """policy 평가 summary, episode별 결과, step trace, action 분포를 함께 반환한다."""
    rewards = []
    unmet_values = []
    rejected_return_values = []
    movement_cost_values = []
    service_rates = []
    episode_rows = []
    step_rows = []
    action_counts = {station_id: 0 for station_id in range(env.config.station_count)}
    same_location_count = 0
    total_steps = 0
    for episode in range(episodes):
        options = (
            {"daily_index": _evaluation_daily_index(episode, episodes, len(env.config.daily_dates))}
            if sequential_dates and env.config.daily_dates
            else None
        )
        _, reset_info = env.reset(seed=seed + episode, options=options)
        done = False
        reward_sum = 0.0
        served_sum = 0
        unmet_sum = 0
        rejected_return_sum = 0
        movement_cost_sum = 0.0
        episode_same_location_count = 0
        active_date = reset_info.get("active_date") or "-"
        while not done:
            previous_location = env.truck_location
            action = int(policy.act(env))
            if action == previous_location:
                same_location_count += 1
                episode_same_location_count += 1
            action_counts[action] += 1
            _, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward_sum += reward
            served_sum += int(info["served_demand"])
            unmet_sum += int(info["unmet_demand"])
            rejected_return_sum += int(info["rejected_returns"])
            movement_cost_sum += float(info["movement_cost"])
            total_steps += 1
            step_rows.append(
                {
                    "policy": label,
                    "episode": episode + 1,
                    "date": active_date,
                    "time_step": int(info["time_step"]),
                    "action": action,
                    "previous_truck_location": previous_location,
                    "truck_location": int(info["truck_location"]),
                    "truck_bikes": int(info["truck_bikes"]),
                    "reward": float(reward),
                    "served_demand": int(info["served_demand"]),
                    "unmet_demand": int(info["unmet_demand"]),
                    "rejected_returns": int(info["rejected_returns"]),
                    "movement_cost": float(info["movement_cost"]),
                    "moved_bikes": int(info["moved_bikes"]),
                    "station_bikes": "|".join(str(value) for value in info["station_bikes"]),
                    "demand": "|".join(str(value) for value in info["demand"]),
                    "returns": "|".join(str(value) for value in info["returns"]),
                }
            )
        total_demand = served_sum + unmet_sum
        rewards.append(reward_sum)
        unmet_values.append(unmet_sum)
        rejected_return_values.append(rejected_return_sum)
        movement_cost_values.append(movement_cost_sum)
        service_rates.append(served_sum / total_demand if total_demand else 1.0)
        episode_rows.append(
            {
                "policy": label,
                "episode": episode + 1,
                "date": active_date,
                "reward": reward_sum,
                "served_demand": served_sum,
                "unmet_demand": unmet_sum,
                "rejected_returns": rejected_return_sum,
                "movement_cost": movement_cost_sum,
                "service_rate": service_rates[-1],
                "same_location_steps": episode_same_location_count,
            }
        )
        if verbose:
            print(
                f"[{label}] episode {episode + 1:02d}/{episodes} "
                f"date={active_date} reward={reward_sum:.1f} "
                f"unmet={unmet_sum} rejected={rejected_return_sum} "
                f"service_rate={service_rates[-1]:.3f}"
            )
    return {
        "summary": {
            "avg_reward": float(np.mean(rewards)),
            "avg_unmet_demand": float(np.mean(unmet_values)),
            "avg_rejected_returns": float(np.mean(rejected_return_values)),
            "avg_movement_cost": float(np.mean(movement_cost_values)),
            "avg_service_rate": float(np.mean(service_rates)),
            "same_location_rate": same_location_count / total_steps if total_steps else 0.0,
        },
        "episodes": episode_rows,
        "steps": step_rows,
        "action_counts": action_counts,
    }


def save_model(policy: GreedyQPolicy, path: Path) -> None:
    """학습된 Q-policy의 network weight를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.network.state_dict(), path)


def _evaluation_daily_index(episode: int, episode_count: int, day_count: int) -> int:
    """평가 episode가 전체 날짜에 고르게 분포하도록 daily index를 고른다."""
    if day_count <= 0:
        raise ValueError("day_count must be positive")
    if episode_count <= 1:
        return 0
    return round(episode * (day_count - 1) / (episode_count - 1))


def _average_metric(metrics: list[dict[str, float]], key: str) -> float:
    """metric list에서 특정 key의 평균을 안전하게 계산한다."""
    if not metrics:
        return 0.0
    return float(np.mean([metric.get(key, 0.0) for metric in metrics]))


def _progress_bar(current: int, total: int, width: int = 12) -> str:
    """현재 episode 진행률을 ASCII progress bar로 만든다."""
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = min(width, max(0, round(width * current / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"
