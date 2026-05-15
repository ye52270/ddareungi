"""V2 PyTorch DQN 학습 CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean

from ddareungi_rl.agents.dqn import Transition
from ddareungi_rl.agents.torch_dqn import TorchDQNAgent, TorchDQNConfig
from ddareungi_rl.envs import ToyDdareungiEnv
from ddareungi_rl.training.evaluate import EpisodeResult, save_episode_log, service_rate


@dataclass
class TorchTrainingEpisodeMetric:
    """PyTorch DQN 학습 episode 하나의 요약 지표를 저장한다."""

    episode: int
    episode_reward: float
    unmet_demand: int
    service_rate: float
    movement_cost: int
    epsilon: float
    loss_mean: float | None


def train_torch_dqn(
    episodes: int,
    seed: int,
    config: TorchDQNConfig,
    device: str | None = None,
) -> tuple[TorchDQNAgent, list[TorchTrainingEpisodeMetric], EpisodeResult]:
    """ToyDdareungiEnv에서 PyTorch DQN을 학습하고 metrics와 마지막 episode log를 반환한다."""
    env = ToyDdareungiEnv(seed=seed)
    agent = TorchDQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_space_n,
        config=config,
        seed=seed,
        device=device,
    )
    global_step = 0
    metrics: list[TorchTrainingEpisodeMetric] = []
    last_result: EpisodeResult | None = None

    for episode in range(episodes):
        episode_seed = seed + episode
        state, _ = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0.0
        total_served = 0
        total_unmet = 0
        total_full_returns = 0
        total_movement_cost = 0
        action_counts = {station_id: 0 for station_id in range(env.action_space_n)}
        same_location_actions = 0
        losses: list[float] = []
        log: list[dict[str, object]] = [
            {
                "event": "reset",
                "state": state,
                "info": env.last_info.copy(),
            }
        ]

        while not done:
            epsilon = agent.epsilon(global_step)
            action = agent.select_action(state, epsilon)
            previous_location = env.truck_location
            action_counts[action] = action_counts.get(action, 0) + 1
            if action == previous_location:
                same_location_actions += 1

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(Transition(state, action, reward, next_state, done))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            episode_reward += reward
            total_served += int(info["served_demand"])
            total_unmet += int(info["unmet_demand"])
            total_full_returns += int(info["full_returns"])
            total_movement_cost += int(info["movement_cost"])
            total_demand = total_served + total_unmet
            info["policy_name"] = "torch-dqn"
            info["learning_stage"] = "PyTorch DQN 학습 중"
            info["episode_reward_so_far"] = episode_reward
            info["episode_served_demand_so_far"] = total_served
            info["episode_unmet_demand_so_far"] = total_unmet
            info["episode_total_demand_so_far"] = total_demand
            info["episode_full_returns_so_far"] = total_full_returns
            info["episode_movement_cost_so_far"] = total_movement_cost
            info["service_rate_so_far"] = service_rate(total_served, total_demand)
            info["reward_formula"] = "-10 * unmet_demand - movement_cost"
            info["epsilon"] = epsilon
            log.append(
                {
                    "event": "step",
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
            )
            state = next_state
            global_step += 1

        total_demand = total_served + total_unmet
        episode_metric = TorchTrainingEpisodeMetric(
            episode=episode + 1,
            episode_reward=episode_reward,
            unmet_demand=total_unmet,
            service_rate=service_rate(total_served, total_demand),
            movement_cost=total_movement_cost,
            epsilon=agent.epsilon(global_step),
            loss_mean=mean(losses) if losses else None,
        )
        metrics.append(episode_metric)
        last_result = EpisodeResult(
            episode_reward=episode_reward,
            served_demand=total_served,
            unmet_demand=total_unmet,
            total_demand=total_demand,
            full_returns=total_full_returns,
            movement_cost=total_movement_cost,
            steps=len(log) - 1,
            log=log,
            action_counts=action_counts,
            same_location_actions=same_location_actions,
        )

    env.close()
    if last_result is None:
        raise ValueError("episodes must be at least 1")
    agent.update_target_network()
    return agent, metrics, last_result


def save_torch_metrics(metrics: list[TorchTrainingEpisodeMetric], output_path: Path) -> None:
    """PyTorch DQN 학습 episode metrics를 JSON 파일로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(metric) for metric in metrics]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_config(args: argparse.Namespace) -> TorchDQNConfig:
    """CLI argument에서 TorchDQNConfig를 만든다."""
    return TorchDQNConfig(
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )


def parse_args() -> argparse.Namespace:
    """PyTorch DQN 학습용 command-line argument를 파싱한다."""
    parser = argparse.ArgumentParser(description="Train a PyTorch DQN policy on ToyDdareungiEnv.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=5000)
    parser.add_argument("--min-replay-size", type=int, default=100)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-out", type=Path, default=Path("outputs/models/torch_dqn_v1.pt"))
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("outputs/metrics/torch_dqn_train_metrics.json"),
    )
    parser.add_argument("--save-log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """PyTorch DQN 학습을 실행하고 model/metrics/log를 저장한다."""
    args = parse_args()
    config = build_config(args)
    agent, metrics, last_result = train_torch_dqn(
        episodes=args.episodes,
        seed=args.seed,
        config=config,
        device=args.device,
    )
    agent.save(args.model_out)
    save_torch_metrics(metrics, args.metrics_out)
    if args.save_log is not None:
        save_episode_log(last_result, args.save_log)

    last_metric = metrics[-1]
    print("PyTorch DQN training summary")
    print(f"episodes={args.episodes}")
    print(f"model_out={args.model_out}")
    print(f"metrics_out={args.metrics_out}")
    print(f"training_last_reward={last_metric.episode_reward:.2f}")
    print(f"training_last_unmet_demand={last_metric.unmet_demand}")
    print(f"training_last_service_rate={last_metric.service_rate:.2%}")
    print(f"training_last_loss_mean={last_metric.loss_mean}")
    print("주의: training_last_* 값은 학습 중 epsilon-greedy 결과이며 성능 비교용이 아닙니다.")


if __name__ == "__main__":
    main()
