"""따릉이 RL 최소 실험을 실행하는 콘솔 메뉴."""

from __future__ import annotations

from pathlib import Path

from ddareungi_rl.baselines import DemandAwarePolicy, LowStockPolicy, NoOpPolicy, RandomPolicy
from ddareungi_rl.data_profile import load_profile, profile_summary
from ddareungi_rl.dqn import (
    DQNConfig,
    evaluate_policy,
    evaluate_policy_with_trace,
    save_model,
    train_dqn,
)
from ddareungi_rl.env import DdareungiEnv
from ddareungi_rl.experiment_log import append_dqn_experiment_log
from ddareungi_rl.reporting import (
    save_baseline_vs_dqn_csv,
    save_experiment_config,
    save_mdp_summary,
    save_policy_trace_reports,
    save_training_history_csv,
)
from ddareungi_rl.visualization import (
    DQN_COMPARISON_CHART_PATH,
    save_baseline_comparison_chart,
    save_action_distribution_chart,
    save_dqn_training_curve,
)


DEFAULT_PROFILE_PATH = Path("outputs/data/magok_3station_daily_profile.json")
FALLBACK_PROFILE_PATH = Path("outputs/data/magok_3station_profile.json")
DEFAULT_MODEL_PATH = Path("outputs/models/simple_dqn.pt")
BASELINE_EPISODES = 30
DQN_TRAINING_EPISODES = 1000


def current_profile_path() -> Path:
    """현재 실행에서 사용할 real-data profile 경로를 반환한다."""
    return DEFAULT_PROFILE_PATH if DEFAULT_PROFILE_PATH.exists() else FALLBACK_PROFILE_PATH


def make_env() -> DdareungiEnv:
    """현재 프로젝트의 기본 실험 환경인 real-data profile 환경을 만든다."""
    profile_path = current_profile_path()
    if not profile_path.exists():
        raise FileNotFoundError(
            "real-data profile이 없습니다. 메뉴 3번을 눌러 생성 명령을 확인하세요."
        )
    config = load_profile(profile_path)
    return DdareungiEnv(config=config, seed=42)


def run_baselines() -> None:
    """real-data profile 환경에서 baseline policy들을 평가하고 출력한다."""
    env = make_env()
    results: dict[str, dict[str, float]] = {}
    policies = {
        "no-op": NoOpPolicy(),
        "random": RandomPolicy(seed=42),
        "low-stock": LowStockPolicy(),
        "demand-aware": DemandAwarePolicy(),
    }
    for name, policy in policies.items():
        print()
        print(f"== {name} baseline ==")
        result = evaluate_policy(
            env,
            policy,
            episodes=BASELINE_EPISODES,
            verbose=True,
            label=name,
        )
        results[name] = result
        print(f"{name}: {result}")
    _print_baseline_chart_result(results)


def _print_baseline_chart_result(results: dict[str, dict[str, float]]) -> None:
    """baseline 비교 그래프 저장 결과를 콘솔에 출력한다."""
    try:
        chart_path = save_baseline_comparison_chart(results)
    except RuntimeError as exc:
        print()
        print(f"baseline 비교 그래프 저장 실패: {exc}")
        return
    print()
    print(f"baseline 비교 그래프 저장: {chart_path}")


def run_training() -> None:
    """real-data profile 환경에서 DQN을 학습하고 평가 결과를 출력한다."""
    profile_path = current_profile_path()
    env = make_env()
    config = DQNConfig(episodes=DQN_TRAINING_EPISODES)
    policy, metrics = train_dqn(env, config=config, verbose=True)
    save_model(policy, DEFAULT_MODEL_PATH)
    dqn_report = evaluate_policy_with_trace(
        env,
        policy,
        episodes=BASELINE_EPISODES,
        verbose=True,
        label="dqn",
    )
    result = dqn_report["summary"]
    print(f"last_training_metric={metrics[-1]}")
    print(f"dqn_eval={result}")
    print(f"model_saved={DEFAULT_MODEL_PATH}")
    baseline_results = _evaluate_baseline_summary(env)
    curve_path = _print_dqn_curve_result(metrics, baseline_results)
    _print_dqn_comparison_chart(baseline_results, result)
    _save_report_outputs(
        env=env,
        config=config,
        profile_path=profile_path,
        metrics=metrics,
        baseline_results=baseline_results,
        dqn_report=dqn_report,
    )
    log_path = append_dqn_experiment_log(
        config=config,
        env=env,
        eval_episodes=BASELINE_EPISODES,
        eval_result=result,
        last_training_metric=metrics[-1],
        model_path=DEFAULT_MODEL_PATH,
        curve_path=curve_path,
        profile_path=profile_path,
    )
    print(f"dqn_experiment_log_saved={log_path}")


def _save_report_outputs(
    *,
    env: DdareungiEnv,
    config: DQNConfig,
    profile_path: Path,
    metrics: list[dict[str, float]],
    baseline_results: dict[str, dict[str, float]],
    dqn_report: dict[str, object],
) -> None:
    """논문식 결과표와 trace CSV를 outputs/reports에 저장하고 경로를 출력한다."""
    comparison_results = dict(baseline_results)
    comparison_results["dqn"] = dqn_report["summary"]  # type: ignore[index]
    report_paths = [
        save_experiment_config(env=env, config=config, profile_path=profile_path),
        save_mdp_summary(env),
        save_baseline_vs_dqn_csv(comparison_results),
        save_training_history_csv(metrics),
    ]
    trace_paths = save_policy_trace_reports(
        policy_name="dqn",
        station_names=list(env.config.station_names),
        evaluation_report=dqn_report,
    )
    try:
        action_chart = save_action_distribution_chart(
            dqn_report["action_counts"],  # type: ignore[index]
            list(env.config.station_names),
        )
    except RuntimeError as exc:
        print(f"action 분포 그래프 저장 실패: {exc}")
        action_chart = None

    for path in report_paths:
        print(f"report_saved={path}")
    for path in trace_paths.values():
        print(f"report_saved={path}")
    if action_chart:
        print(f"action_distribution_chart_saved={action_chart}")


def _print_dqn_comparison_chart(
    baseline_results: dict[str, dict[str, float]],
    dqn_result: dict[str, float],
) -> Path | None:
    """baseline과 DQN 평가 결과를 한 그래프로 저장하고 경로를 출력한다."""
    comparison_results = dict(baseline_results)
    comparison_results["dqn"] = dqn_result
    try:
        chart_path = save_baseline_comparison_chart(
            comparison_results,
            output_path=DQN_COMPARISON_CHART_PATH,
            title="따릉이 Baseline vs DQN 비교",
        )
    except RuntimeError as exc:
        print()
        print(f"DQN 비교 그래프 저장 실패: {exc}")
        return None
    print(f"dqn_comparison_chart_saved={chart_path}")
    return chart_path


def _evaluate_baseline_summary(env: DdareungiEnv) -> dict[str, dict[str, float]]:
    """DQN과 비교할 baseline 결과를 같은 평가 조건에서 조용히 계산한다."""
    policies = {
        "no-op": NoOpPolicy(),
        "random": RandomPolicy(seed=42),
        "low-stock": LowStockPolicy(),
        "demand-aware": DemandAwarePolicy(),
    }
    return {
        name: evaluate_policy(
            env,
            policy,
            episodes=BASELINE_EPISODES,
            verbose=False,
            label=name,
        )
        for name, policy in policies.items()
    }


def _print_dqn_curve_result(
    metrics: list[dict[str, float]],
    baseline_results: dict[str, dict[str, float]],
) -> Path | None:
    """DQN 학습 곡선 저장 결과를 콘솔에 출력한다."""
    low_stock_reward = baseline_results.get("low-stock", {}).get("avg_reward")
    try:
        chart_path = save_dqn_training_curve(
            metrics,
            baseline_reward=low_stock_reward,
            baseline_label="low-stock",
        )
    except RuntimeError as exc:
        print()
        print(f"DQN 학습 곡선 저장 실패: {exc}")
        return None
    print(f"dqn_training_curve_saved={chart_path}")
    return chart_path


def print_profile_help() -> None:
    """현재 profile 파일 상태와 생성 명령 예시를 출력한다."""
    profile_path = current_profile_path()
    print()
    print("== 데이터 profile 상태 ==")
    if profile_path.exists():
        summary = profile_summary(profile_path)
        metadata = summary["metadata"]
        print(f"사용 파일: {profile_path}")
        print(f"대여소: {', '.join(summary['stations'])}")
        print(f"날짜 수: {metadata.get('day_count', '-')}")
        print(f"정규화: {metadata.get('normalization', '-')}")
    else:
        print("사용 가능한 profile 파일이 아직 없습니다.")

    print()
    print("== daily profile 생성 명령 예시 ==")
    print(
        "ddareungi-build-profile \\\n"
        "  --rental-dir \"data/서울특별시 공공자전거 대여이력 정보_2025\" \\\n"
        "  --station-keyword \"마곡\" \\\n"
        "  --station-count 3 \\\n"
        "  --profile-kind daily \\\n"
        "  --max-sample-high 10 \\\n"
        "  --output outputs/data/magok_3station_daily_profile.json"
    )


def print_menu() -> None:
    """사용자가 선택할 수 있는 최소 메뉴를 출력한다."""
    print()
    print("Ddareungi RL")
    print("1. Baseline 평가")
    print("2. DQN 학습/평가")
    print("3. 데이터 profile 상태/생성 안내")
    print("0. 종료")


def main() -> None:
    """콘솔 메뉴를 반복 실행한다."""
    while True:
        print_menu()
        choice = input("선택: ").strip()
        if choice == "1":
            run_baselines()
        elif choice == "2":
            run_training()
        elif choice == "3":
            print_profile_help()
        elif choice == "0":
            print("종료합니다.")
            return
        else:
            print("알 수 없는 선택입니다.")


if __name__ == "__main__":
    main()
