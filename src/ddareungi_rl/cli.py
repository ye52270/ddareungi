"""따릉이 RL 최소 실험을 실행하는 콘솔 메뉴."""

from __future__ import annotations

import csv
from pathlib import Path

from ddareungi_rl.baselines import DemandAwarePolicy, LowStockPolicy, NoOpPolicy, RandomPolicy
from ddareungi_rl.dashboard import save_experiment_dashboard
from ddareungi_rl.data_profile import load_profile, profile_summary
from ddareungi_rl.dqn import (
    DQNConfig,
    evaluate_policy,
    evaluate_policy_with_trace,
    save_model,
    train_double_dqn,
    train_dqn,
    train_dueling_dqn,
)
from ddareungi_rl.env import DdareungiEnv
from ddareungi_rl.experiment_log import append_dqn_experiment_log
from ddareungi_rl.reporting import (
    save_algorithm_comparison_from_reports,
    save_baseline_vs_dqn_csv,
    save_experiment_config,
    save_mdp_summary,
    save_multiseed_reports,
    save_policy_trace_reports,
    save_training_history_csv,
)
from ddareungi_rl.visualization import (
    DQN_COMPARISON_CHART_PATH,
    save_algorithm_comparison_chart,
    save_baseline_comparison_chart,
    save_action_distribution_chart,
    save_dqn_training_curve,
    save_multiseed_summary_chart,
)


DEFAULT_PROFILE_PATH = Path("outputs/data/magok_3station_daily_profile.json")
FALLBACK_PROFILE_PATH = Path("outputs/data/magok_3station_profile.json")
DEFAULT_MODEL_PATH = Path("outputs/models/simple_dqn.pt")
DOUBLE_MODEL_PATH = Path("outputs/models/double_dqn.pt")
DUELING_MODEL_PATH = Path("outputs/models/dueling_dqn.pt")
BASELINE_EPISODES = 30
DQN_TRAINING_EPISODES = 1000
DQN_MULTI_SEEDS = (42, 142, 242, 342, 442)


def current_profile_path() -> Path:
    """현재 실행에서 사용할 real-data profile 경로를 반환한다."""
    return DEFAULT_PROFILE_PATH if DEFAULT_PROFILE_PATH.exists() else FALLBACK_PROFILE_PATH


def make_env() -> DdareungiEnv:
    """현재 프로젝트의 기본 실험 환경인 real-data profile 환경을 만든다."""
    profile_path = current_profile_path()
    if not profile_path.exists():
        raise FileNotFoundError(
            "real-data profile이 없습니다. 데이터 profile 상태/생성 안내 메뉴를 확인하세요."
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
    _run_q_algorithm(
        algorithm_name="dqn",
        model_path=DEFAULT_MODEL_PATH,
        train_fn=train_dqn,
        comparison_chart_path=DQN_COMPARISON_CHART_PATH,
        training_curve_path=Path("outputs/figures/dqn_training_curve.png"),
        comparison_csv_path=Path("outputs/reports/baseline_vs_dqn.csv"),
        training_history_path=Path("outputs/reports/dqn_training_history.csv"),
        verbose_eval=True,
    )


def run_dueling_training() -> None:
    """real-data profile 환경에서 Dueling DQN을 학습하고 평가 결과를 출력한다."""
    _run_q_algorithm(
        algorithm_name="dueling_dqn",
        model_path=DUELING_MODEL_PATH,
        train_fn=train_dueling_dqn,
        comparison_chart_path=Path("outputs/figures/dueling_dqn_vs_baseline_comparison.png"),
        training_curve_path=Path("outputs/figures/dueling_dqn_training_curve.png"),
        comparison_csv_path=Path("outputs/reports/dueling_dqn_vs_baseline.csv"),
        training_history_path=Path("outputs/reports/dueling_dqn_training_history.csv"),
        verbose_eval=True,
    )


def run_double_training() -> None:
    """real-data profile 환경에서 Double DQN을 학습하고 평가 결과를 출력한다."""
    _run_q_algorithm(
        algorithm_name="double_dqn",
        model_path=DOUBLE_MODEL_PATH,
        train_fn=train_double_dqn,
        comparison_chart_path=Path("outputs/figures/double_dqn_vs_baseline_comparison.png"),
        training_curve_path=Path("outputs/figures/double_dqn_training_curve.png"),
        comparison_csv_path=Path("outputs/reports/double_dqn_vs_baseline.csv"),
        training_history_path=Path("outputs/reports/double_dqn_training_history.csv"),
        verbose_eval=True,
    )


def _run_q_algorithm(
    *,
    algorithm_name: str,
    model_path: Path,
    train_fn: object,
    comparison_chart_path: Path,
    training_curve_path: Path,
    comparison_csv_path: Path,
    training_history_path: Path,
    verbose_eval: bool,
) -> None:
    """DQN 계열 알고리즘을 학습/평가하고 공통 report를 저장한다."""
    profile_path = current_profile_path()
    env = make_env()
    config = DQNConfig(episodes=DQN_TRAINING_EPISODES)
    policy, metrics = train_fn(env, config=config, verbose=True)  # type: ignore[operator]
    save_model(policy, model_path)
    dqn_report = evaluate_policy_with_trace(
        env,
        policy,
        episodes=BASELINE_EPISODES,
        verbose=verbose_eval,
        label=algorithm_name,
    )
    result = dqn_report["summary"]
    print(f"last_training_metric={metrics[-1]}")
    print(f"{algorithm_name}_eval={result}")
    print(f"model_saved={model_path}")
    baseline_results = _evaluate_baseline_summary(env)
    curve_path = _print_dqn_curve_result(
        metrics,
        baseline_results,
        algorithm_name=algorithm_name,
        output_path=training_curve_path,
        baseline_label="low-stock",
    )
    _print_dqn_comparison_chart(
        baseline_results,
        result,
        algorithm_name=algorithm_name,
        output_path=comparison_chart_path,
    )
    _save_report_outputs(
        env=env,
        config=config,
        profile_path=profile_path,
        metrics=metrics,
        baseline_results=baseline_results,
        dqn_report=dqn_report,
        algorithm_name=algorithm_name,
        comparison_csv_path=comparison_csv_path,
        training_history_path=training_history_path,
    )
    if algorithm_name == "dqn":
        log_path = append_dqn_experiment_log(
            config=config,
            env=env,
            eval_episodes=BASELINE_EPISODES,
            eval_result=result,
            last_training_metric=metrics[-1],
            model_path=model_path,
            curve_path=curve_path,
            profile_path=profile_path,
        )
        print(f"dqn_experiment_log_saved={log_path}")


def run_multiseed_training() -> None:
    """여러 seed에서 DQN을 반복 학습/평가해 안정성을 확인한다."""
    config = DQNConfig(episodes=DQN_TRAINING_EPISODES)
    rows: list[dict[str, float]] = []
    print()
    print(f"== DQN multi-seed 평가: seeds={DQN_MULTI_SEEDS} ==")
    for seed in DQN_MULTI_SEEDS:
        print()
        print(f"[multi-seed] seed={seed} 학습 시작")
        env = make_env()
        policy, _ = train_dqn(env, config=config, seed=seed, verbose=True, log_interval=100)
        result = evaluate_policy(
            env,
            policy,
            episodes=BASELINE_EPISODES,
            seed=1000,
            verbose=False,
            label=f"dqn-seed-{seed}",
        )
        row = {"seed": float(seed), **result}
        rows.append(row)
        print(
            f"[multi-seed] seed={seed} "
            f"reward={result['avg_reward']:.2f} "
            f"unmet={result['avg_unmet_demand']:.2f} "
            f"rejected={result['avg_rejected_returns']:.2f} "
            f"service={result['avg_service_rate']:.3f}"
        )
    report_paths = save_multiseed_reports(rows)
    chart_path = save_multiseed_summary_chart(rows)
    print()
    print(f"multiseed_runs_saved={report_paths['runs']}")
    print(f"multiseed_summary_saved={report_paths['summary']}")
    print(f"multiseed_chart_saved={chart_path}")


def run_algorithm_comparison() -> None:
    """이미 실행된 DQN 계열 결과를 모아 알고리즘별 비교표와 그래프를 만든다."""
    comparison_path = save_algorithm_comparison_from_reports()
    with comparison_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        print("비교할 알고리즘 결과가 없습니다. 먼저 DQN/Double/Dueling을 실행하세요.")
        return
    chart_path = save_algorithm_comparison_chart(rows)
    print()
    print("== 알고리즘 결과 비교 ==")
    for row in rows:
        print(
            f"{row['algorithm']}: "
            f"reward={float(row['avg_reward']):.2f}, "
            f"unmet={float(row['avg_unmet_demand']):.2f}, "
            f"rejected={float(row['avg_rejected_returns']):.2f}, "
            f"service={float(row['avg_service_rate']):.3f}"
        )
    print(f"algorithm_comparison_saved={comparison_path}")
    print(f"algorithm_comparison_chart_saved={chart_path}")


def _save_report_outputs(
    *,
    env: DdareungiEnv,
    config: DQNConfig,
    profile_path: Path,
    metrics: list[dict[str, float]],
    baseline_results: dict[str, dict[str, float]],
    dqn_report: dict[str, object],
    algorithm_name: str,
    comparison_csv_path: Path,
    training_history_path: Path,
) -> None:
    """논문식 결과표와 trace CSV를 outputs/reports에 저장하고 경로를 출력한다."""
    comparison_results = dict(baseline_results)
    comparison_results[algorithm_name] = dqn_report["summary"]  # type: ignore[index]
    report_paths = [
        save_experiment_config(env=env, config=config, profile_path=profile_path),
        save_mdp_summary(env),
        save_baseline_vs_dqn_csv(comparison_results, output_path=comparison_csv_path),
        save_training_history_csv(metrics, output_path=training_history_path),
    ]
    trace_paths = save_policy_trace_reports(
        policy_name=algorithm_name,
        station_names=list(env.config.station_names),
        evaluation_report=dqn_report,
    )
    try:
        action_chart = save_action_distribution_chart(
            dqn_report["action_counts"],  # type: ignore[index]
            list(env.config.station_names),
            output_path=Path(f"outputs/figures/{algorithm_name}_action_distribution.png"),
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
    dashboard_path = save_experiment_dashboard(
        algorithm_name=algorithm_name,
        comparison_path=comparison_csv_path,
        training_history_path=training_history_path,
        action_distribution_path=trace_paths["action_distribution"],
        evaluation_episodes_path=trace_paths["evaluation_episodes"],
    )
    print(f"dashboard_saved={dashboard_path}")


def _print_dqn_comparison_chart(
    baseline_results: dict[str, dict[str, float]],
    dqn_result: dict[str, float],
    algorithm_name: str = "dqn",
    output_path: Path = DQN_COMPARISON_CHART_PATH,
) -> Path | None:
    """baseline과 DQN 평가 결과를 한 그래프로 저장하고 경로를 출력한다."""
    comparison_results = dict(baseline_results)
    comparison_results[algorithm_name] = dqn_result
    try:
        chart_path = save_baseline_comparison_chart(
            comparison_results,
            output_path=output_path,
            title=f"따릉이 Baseline vs {algorithm_name} 비교",
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
    algorithm_name: str = "dqn",
    output_path: Path = Path("outputs/figures/dqn_training_curve.png"),
    baseline_label: str = "low-stock",
) -> Path | None:
    """DQN 계열 학습 곡선 저장 결과를 콘솔에 출력한다."""
    low_stock_reward = baseline_results.get("low-stock", {}).get("avg_reward")
    try:
        chart_path = save_dqn_training_curve(
            metrics,
            output_path=output_path,
            baseline_reward=low_stock_reward,
            baseline_label=baseline_label,
            algorithm_label=_algorithm_display_name(algorithm_name),
        )
    except RuntimeError as exc:
        print()
        print(f"DQN 학습 곡선 저장 실패: {exc}")
        return None
    print(f"{algorithm_name}_training_curve_saved={chart_path}")
    return chart_path


def _algorithm_display_name(algorithm_name: str) -> str:
    """파일/코드용 알고리즘 이름을 그래프 제목용 표시 이름으로 바꾼다."""
    names = {
        "dqn": "DQN",
        "double_dqn": "Double DQN",
        "dueling_dqn": "Dueling DQN",
    }
    return names.get(algorithm_name, algorithm_name.replace("_", " ").title())


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
    print("3. Double DQN 학습/평가")
    print("4. Dueling DQN 학습/평가")
    print("5. 데이터 profile 상태/생성 안내")
    print("6. DQN multi-seed 평가")
    print("7. 알고리즘 결과 비교")
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
            run_double_training()
        elif choice == "4":
            run_dueling_training()
        elif choice == "5":
            print_profile_help()
        elif choice == "6":
            run_multiseed_training()
        elif choice == "7":
            run_algorithm_comparison()
        elif choice == "0":
            print("종료합니다.")
            return
        else:
            print("알 수 없는 선택입니다.")


if __name__ == "__main__":
    main()
