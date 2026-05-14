"""V0/V1 policy를 같은 metric으로 평가한다."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Literal

from ddareungi_rl.agents import DQNPolicy
from ddareungi_rl.envs import ToyDdareungiEnv
from ddareungi_rl.policies import LowStockPolicy, RandomPolicy
from ddareungi_rl.policies.baselines import Policy


PolicyName = Literal["random", "low-stock", "dqn"]
RenderChoice = Literal["none", "ansi", "human"]


@dataclass
class EpisodeResult:
    """평가된 episode 하나의 요약 지표와 replay log를 보관한다."""

    episode_reward: float
    served_demand: int
    unmet_demand: int
    total_demand: int
    full_returns: int
    movement_cost: int
    steps: int
    log: list[dict[str, object]]


def make_policy(
    name: PolicyName,
    seed: int | None = None,
    model_path: Path | None = None,
) -> Policy:
    """CLI policy 이름으로 평가용 policy 인스턴스를 만든다."""
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "low-stock":
        return LowStockPolicy()
    if name == "dqn":
        if model_path is None:
            raise ValueError("--model-path is required when --policy dqn")
        return DQNPolicy(model_path)
    raise ValueError(f"Unknown policy: {name}")


def run_episode(
    env: ToyDdareungiEnv,
    policy: Policy,
    seed: int | None = None,
    render_mode: RenderChoice = "none",
    policy_name: str = "baseline",
) -> EpisodeResult:
    """episode 하나를 끝까지 실행하고 선택적으로 렌더링하며 replay record를 모은다."""
    state, info = env.reset(seed=seed)
    done = False
    episode_reward = 0.0
    total_served = 0
    total_unmet = 0
    total_demand = 0
    total_full_returns = 0
    total_movement_cost = 0
    log: list[dict[str, object]] = [
        {
            "event": "reset",
            "state": state,
            "info": info,
        }
    ]

    if render_mode in ("ansi", "human"):
        print(env.render())

    while not done:
        action = policy.select_action(env)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        total_served += int(step_info["served_demand"])
        total_unmet += int(step_info["unmet_demand"])
        total_demand = total_served + total_unmet
        total_full_returns += int(step_info["full_returns"])
        total_movement_cost += int(step_info["movement_cost"])
        step_info["policy_name"] = policy_name
        step_info["learning_stage"] = (
            "DQN greedy 평가" if policy_name == "dqn" else "학습 전 기준 정책"
        )
        step_info["episode_reward_so_far"] = episode_reward
        step_info["episode_served_demand_so_far"] = total_served
        step_info["episode_unmet_demand_so_far"] = total_unmet
        step_info["episode_total_demand_so_far"] = total_demand
        step_info["episode_full_returns_so_far"] = total_full_returns
        step_info["episode_movement_cost_so_far"] = total_movement_cost
        step_info["service_rate_so_far"] = service_rate(total_served, total_demand)
        step_info["reward_formula"] = "-10 * unmet_demand - movement_cost"

        log.append(
            {
                "event": "step",
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "terminated": terminated,
                "truncated": truncated,
                "info": step_info,
            }
        )
        state = next_state

        if render_mode == "ansi":
            print(env.render())

    return EpisodeResult(
        episode_reward=episode_reward,
        served_demand=total_served,
        unmet_demand=total_unmet,
        total_demand=total_demand,
        full_returns=total_full_returns,
        movement_cost=total_movement_cost,
        steps=len(log) - 1,
        log=log,
    )


def service_rate(served_demand: int, total_demand: int) -> float:
    """전체 수요 중 처리된 수요 비율을 반환한다."""
    if total_demand == 0:
        return 1.0
    return served_demand / total_demand


def evaluate(
    policy_name: PolicyName,
    episodes: int,
    seed: int,
    render_mode: RenderChoice,
    model_path: Path | None = None,
) -> list[EpisodeResult]:
    """하나의 policy를 여러 seed episode로 평가한다."""
    env_render_mode = "human" if render_mode == "human" else None
    env = ToyDdareungiEnv(render_mode=env_render_mode, seed=seed)
    policy = make_policy(policy_name, seed=seed, model_path=model_path)

    results = []
    for episode in range(episodes):
        episode_seed = seed + episode
        episode_render_mode = render_mode if episode == 0 else "none"
        results.append(
            run_episode(
                env=env,
                policy=policy,
                seed=episode_seed,
                render_mode=episode_render_mode,
                policy_name=policy_name,
            )
        )

    env.close()
    return results


def save_episode_log(result: EpisodeResult, output_path: Path) -> None:
    """episode 하나의 replay log를 보기 좋게 포맷된 JSON으로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """baseline 평가용 command-line argument를 파싱한다."""
    parser = argparse.ArgumentParser(description="Evaluate Ddareungi policies on the toy MDP.")
    parser.add_argument(
        "--policy",
        choices=["random", "low-stock", "dqn"],
        default="random",
        help="Policy to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render-mode",
        choices=["none", "ansi", "human"],
        default="none",
        help="'ansi' prints text frames manually; 'human' lets env.step render.",
    )
    parser.add_argument(
        "--save-log",
        type=Path,
        default=None,
        help="Optional path for the first episode replay log JSON.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Saved DQN model path. Required when --policy dqn.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 평가를 실행하고 집계 지표를 출력한다."""
    args = parse_args()
    results = evaluate(
        policy_name=args.policy,
        episodes=args.episodes,
        seed=args.seed,
        render_mode=args.render_mode,
        model_path=args.model_path,
    )

    if args.save_log is not None:
        save_episode_log(results[0], args.save_log)

    print("Evaluation summary")
    print(f"policy={args.policy}")
    print(f"episodes={args.episodes}")
    print(f"avg_reward={mean(r.episode_reward for r in results):.2f}")
    print(f"avg_unmet_demand={mean(r.unmet_demand for r in results):.2f}")
    print(f"avg_service_rate={mean(service_rate(r.served_demand, r.total_demand) for r in results):.2%}")
    print(f"avg_full_returns={mean(r.full_returns for r in results):.2f}")
    print(f"avg_movement_cost={mean(r.movement_cost for r in results):.2f}")


if __name__ == "__main__":
    main()
