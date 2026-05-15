"""따릉이 RL 실험을 선택해서 실행하는 콘솔 메뉴."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from ddareungi_rl.agents import DQNConfig
from ddareungi_rl.training.evaluate import (
    EpisodeResult,
    aggregate_action_counts,
    evaluate,
    same_location_rate,
    save_episode_log,
    service_rate,
)
from ddareungi_rl.training.train_dqn import save_metrics, train_dqn
from ddareungi_rl.visualization.demo import create_demo_log
from ddareungi_rl.visualization.pixel_replay import load_episode_log
from ddareungi_rl.visualization.pygame_replay import replay_window


DEFAULT_MODEL_PATH = Path("outputs/models/dqn_small.json")
DEFAULT_METRICS_PATH = Path("outputs/metrics/dqn_small_metrics.json")
DEFAULT_TRAIN_LOG_PATH = Path("outputs/logs/dqn_small_train_episode.json")
DEFAULT_EVAL_LOG_PATH = Path("outputs/logs/dqn_small_eval_episode.json")
DEFAULT_BASELINE_LOG_PATH = Path("outputs/logs/baseline_low_stock_episode.json")
DEFAULT_DEMO_LOG_PATH = Path("outputs/demo_episode.json")


def summarize_results(policy_name: str, results: list[EpisodeResult]) -> str:
    """평가 결과 목록을 콘솔 출력용 요약 문자열로 변환한다."""
    return "\n".join(
        [
            f"policy={policy_name}",
            f"episodes={len(results)}",
            f"avg_reward={mean(result.episode_reward for result in results):.2f}",
            f"avg_unmet_demand={mean(result.unmet_demand for result in results):.2f}",
            f"avg_service_rate={mean(service_rate(result.served_demand, result.total_demand) for result in results):.2%}",
            f"avg_full_returns={mean(result.full_returns for result in results):.2f}",
            f"avg_movement_cost={mean(result.movement_cost for result in results):.2f}",
            f"action_distribution={aggregate_action_counts(results)}",
            f"same_location_rate={same_location_rate(results):.2%}",
        ]
    )


def run_baseline_suite(
    episodes: int = 5,
    seed: int = 42,
    save_low_stock_log: Path | None = None,
) -> dict[str, list[EpisodeResult]]:
    """Random, Low-stock, Demand-aware baseline을 같은 seed 묶음에서 평가한다."""
    results = {
        "random": evaluate("random", episodes=episodes, seed=seed, render_mode="none"),
        "low-stock": evaluate("low-stock", episodes=episodes, seed=seed, render_mode="none"),
        "demand-aware": evaluate("demand-aware", episodes=episodes, seed=seed, render_mode="none"),
    }
    if save_low_stock_log is not None:
        save_episode_log(results["low-stock"][0], save_low_stock_log)
    return results


def run_dqn_small_training(
    episodes: int = 100,
    seed: int = 42,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    log_path: Path = DEFAULT_TRAIN_LOG_PATH,
) -> Path:
    """작은 DQN smoke 학습을 실행하고 모델/metrics/log를 저장한다."""
    config = DQNConfig()
    agent, metrics, last_result = train_dqn(episodes=episodes, seed=seed, config=config)
    agent.save(model_path)
    save_metrics(metrics, metrics_path)
    save_episode_log(last_result, log_path)
    last_metric = metrics[-1]
    print("DQN(Small) training complete")
    print(f"episodes={episodes}")
    print(f"model={model_path}")
    print(f"metrics={metrics_path}")
    print(f"training_last_reward={last_metric.episode_reward:.2f}")
    print(f"training_last_unmet_demand={last_metric.unmet_demand}")
    print("주의: training_last_* 값은 학습 중 epsilon-greedy 결과이며 성능 비교용이 아닙니다.")
    return model_path


def run_dqn_small_evaluation(
    episodes: int = 5,
    seed: int = 1000,
    model_path: Path = DEFAULT_MODEL_PATH,
    log_path: Path = DEFAULT_EVAL_LOG_PATH,
) -> list[EpisodeResult]:
    """저장된 DQN(Small) 모델을 held-out seed 묶음에서 greedy 평가한다."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path}가 없습니다. 먼저 메뉴 2번으로 DQN(Small)을 학습하세요."
        )
    results = evaluate(
        "dqn",
        episodes=episodes,
        seed=seed,
        render_mode="none",
        model_path=model_path,
    )
    save_episode_log(results[0], log_path)
    return results


def replay_log(log_path: Path, max_steps: int | None = None) -> None:
    """저장된 episode log를 pygame 창 replay로 실행한다."""
    records = load_episode_log(log_path)
    replay_window(records, max_steps=max_steps)


def run_replay_demo(log_path: Path = DEFAULT_DEMO_LOG_PATH) -> None:
    """Low-stock baseline episode를 생성하고 pygame 창 replay를 실행한다."""
    create_demo_log("low-stock", seed=42, log_path=log_path)
    replay_log(log_path)


def print_menu() -> None:
    """콘솔 메뉴 항목을 출력한다."""
    print()
    print("Ddareungi RL Toy MDP")
    print("1. Baseline 평가(Random + Low-stock + Demand-aware)")
    print("2. DQN(Small) 학습")
    print("3. DQN(Small) 평가")
    print("4. Baseline 평가 + visualization")
    print("5. DQN(Small) 평가 + visualization")
    print("0. 종료")


def run_choice(choice: str) -> bool:
    """메뉴 선택 하나를 실행하고 계속 실행 여부를 반환한다."""
    if choice == "1":
        print("[Baseline 평가] seed=42, episodes=5")
        results_by_policy = run_baseline_suite()
        for policy_name, results in results_by_policy.items():
            print("-" * 40)
            print(summarize_results(policy_name, results))
        return True
    if choice == "2":
        print("[DQN(Small) 학습] seed=42, episodes=100")
        run_dqn_small_training()
        return True
    if choice == "3":
        print("[DQN(Small) 평가] held-out seed=1000, episodes=5")
        try:
            results = run_dqn_small_evaluation()
        except FileNotFoundError as exc:
            print(exc)
            return True
        print(summarize_results("dqn", results))
        return True
    if choice == "4":
        print("[Baseline 평가 + visualization] low-stock 첫 episode replay")
        results_by_policy = run_baseline_suite(save_low_stock_log=DEFAULT_BASELINE_LOG_PATH)
        for policy_name, results in results_by_policy.items():
            print("-" * 40)
            print(summarize_results(policy_name, results))
        print("Visualization 조작: 창을 한 번 클릭한 뒤 Space 일시정지, Right 다음, R 다시보기, Q/Esc 창 닫기")
        print("참고: 메뉴 visualization은 창을 닫으면 프로그램도 종료됩니다.")
        replay_log(DEFAULT_BASELINE_LOG_PATH)
        print("Visualization 창을 닫았습니다. 프로그램을 종료합니다.")
        return False
    if choice == "5":
        print("[DQN(Small) 평가 + visualization] held-out seed=1000, episodes=5")
        try:
            results = run_dqn_small_evaluation()
        except FileNotFoundError as exc:
            print(exc)
            return True
        print(summarize_results("dqn", results))
        print("Visualization 조작: 창을 한 번 클릭한 뒤 Space 일시정지, Right 다음, R 다시보기, Q/Esc 창 닫기")
        print("참고: 메뉴 visualization은 창을 닫으면 프로그램도 종료됩니다.")
        replay_log(DEFAULT_EVAL_LOG_PATH)
        print("Visualization 창을 닫았습니다. 프로그램을 종료합니다.")
        return False
    if choice == "0":
        print("종료합니다.")
        return False
    print("알 수 없는 선택입니다.")
    return True


def main() -> None:
    """사용자 입력을 받아 따릉이 RL 실험 메뉴를 실행한다."""
    running = True
    while running:
        print_menu()
        choice = input("선택: ").strip()
        running = run_choice(choice)


if __name__ == "__main__":
    main()
