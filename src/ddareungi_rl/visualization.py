"""Baseline과 DQN 평가 결과를 비교하는 간단한 시각화 도구."""

from __future__ import annotations

from pathlib import Path


BASELINE_CHART_PATH = Path("outputs/figures/baseline_comparison.png")
DQN_COMPARISON_CHART_PATH = Path("outputs/figures/dqn_vs_baseline_comparison.png")
DQN_TRAINING_CHART_PATH = Path("outputs/figures/dqn_training_curve.png")
ACTION_DISTRIBUTION_CHART_PATH = Path("outputs/figures/action_distribution.png")
DQN_MULTI_SEED_CHART_PATH = Path("outputs/figures/dqn_multiseed_summary.png")
ALGORITHM_COMPARISON_CHART_PATH = Path("outputs/figures/algorithm_comparison.png")


def save_baseline_comparison_chart(
    results: dict[str, dict[str, float]],
    output_path: Path = BASELINE_CHART_PATH,
    title: str = "따릉이 Baseline 정책 비교",
) -> Path:
    """policy별 reward, unmet demand, rejected return, service rate 비교 그래프를 저장한다."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib이 설치되어 있지 않아 그래프를 저장하지 못했습니다. "
            "`pip install -e .` 또는 `pip install matplotlib` 후 다시 실행하세요."
        ) from exc

    _configure_korean_font(plt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policy_names = list(results)
    reward_values = [results[name]["avg_reward"] for name in policy_names]
    unmet_values = [results[name]["avg_unmet_demand"] for name in policy_names]
    rejected_values = [results[name]["avg_rejected_returns"] for name in policy_names]
    service_values = [results[name]["avg_service_rate"] for name in policy_names]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    _draw_bar(axes[0, 0], policy_names, reward_values, "평균 보상", higher_is_better=True)
    _draw_bar(axes[0, 1], policy_names, unmet_values, "미충족 수요", higher_is_better=False)
    _draw_bar(axes[1, 0], policy_names, rejected_values, "반납 실패", higher_is_better=False)
    _draw_bar(axes[1, 1], policy_names, service_values, "서비스율", higher_is_better=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_dqn_training_curve(
    metrics: list[dict[str, float]],
    output_path: Path = DQN_TRAINING_CHART_PATH,
    baseline_reward: float | None = None,
    baseline_label: str = "low-stock baseline",
    algorithm_label: str = "DQN",
) -> Path:
    """DQN episode별 reward, 실패 지표, 이동비용, loss 학습 곡선을 저장한다."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib이 설치되어 있지 않아 그래프를 저장하지 못했습니다. "
            "`pip install -e .` 또는 `pip install matplotlib` 후 다시 실행하세요."
        ) from exc

    _configure_korean_font(plt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [metric["episode"] for metric in metrics]
    rewards = _metric_values(metrics, "reward")
    unmet_values = _metric_values(metrics, "unmet_demand")
    rejected_values = _metric_values(metrics, "rejected_returns")
    movement_values = _metric_values(metrics, "movement_cost")
    loss_values = _metric_values(metrics, "loss")
    reward_average = _moving_average(rewards)
    unmet_average = _moving_average(unmet_values)
    rejected_average = _moving_average(rejected_values)
    movement_average = _moving_average(movement_values)
    loss_average = _moving_average(loss_values)
    best_reward_average = _best_so_far(reward_average)
    recent_reward = _recent_average(rewards)
    recent_unmet = _recent_average(unmet_values)
    recent_rejected = _recent_average(rejected_values)
    recent_movement = _recent_average(movement_values)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(f"{algorithm_label} 학습 추세: 보상은 위로, 실패 지표는 아래로", fontsize=15, fontweight="bold")

    reward_axis = axes[0, 0]
    reward_axis.plot(episodes, rewards, color="#a9b4c2", linewidth=0.8, alpha=0.35, label="episode 보상")
    reward_axis.plot(episodes, reward_average, color="#f2994a", linewidth=2.2, label="최근 20 episode 평균")
    reward_axis.plot(episodes, best_reward_average, color="#2f80ed", linewidth=2, label="최고 이동평균")
    if baseline_reward is not None:
        reward_axis.axhline(
            baseline_reward,
            color="#27ae60",
            linestyle="--",
            linewidth=2,
            label=f"{baseline_label} 평균 보상",
        )
        reward_axis.fill_between(
            episodes,
            baseline_reward,
            max(max(best_reward_average), baseline_reward),
            color="#27ae60",
            alpha=0.08,
        )
    reward_axis.set_title("운영 보상 추세")
    reward_axis.set_ylabel("보상")
    reward_axis.grid(alpha=0.25)
    reward_axis.legend()
    reward_axis.text(
        0.01,
        0.04,
        f"위로 갈수록 좋음 | 최근 100 episode 평균: {recent_reward:.2f}",
        transform=reward_axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#d0d7de"},
    )

    unmet_axis = axes[0, 1]
    unmet_axis.plot(episodes, unmet_values, color="#f4b6b6", linewidth=0.8, alpha=0.45, label="episode 미충족")
    unmet_axis.plot(episodes, unmet_average, color="#eb5757", linewidth=2.2, label="최근 20 episode 평균")
    unmet_axis.set_title("헛걸음 감소 추세")
    unmet_axis.set_ylabel("건수")
    unmet_axis.grid(alpha=0.25)
    unmet_axis.legend()
    unmet_axis.text(
        0.01,
        0.04,
        f"아래로 갈수록 좋음 | 최근 100 episode 평균: {recent_unmet:.2f}건",
        transform=unmet_axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#d0d7de"},
    )

    rejected_axis = axes[1, 0]
    rejected_axis.plot(episodes, rejected_values, color="#f8cfa1", linewidth=0.8, alpha=0.45, label="episode 반납 실패")
    rejected_axis.plot(episodes, rejected_average, color="#f2994a", linewidth=2.2, label="최근 20 episode 평균")
    rejected_axis.set_title("반납 실패 감소 추세")
    rejected_axis.set_xlabel("Episode")
    rejected_axis.set_ylabel("건수")
    rejected_axis.grid(alpha=0.25)
    rejected_axis.legend()
    rejected_axis.text(
        0.01,
        0.86,
        f"아래로 갈수록 좋음 | 최근 100 episode 평균: {recent_rejected:.2f}건",
        transform=rejected_axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#d0d7de"},
    )

    cost_axis = axes[1, 1]
    cost_axis.plot(episodes, movement_values, color="#b8c4d2", linewidth=0.8, alpha=0.45, label="episode 이동비용")
    cost_axis.plot(episodes, movement_average, color="#2f80ed", linewidth=2.2, label="이동비용 평균")
    if any(loss_values):
        cost_axis.plot(episodes, loss_average, color="#9b51e0", linewidth=1.8, alpha=0.9, label="loss 평균")
    cost_axis.set_title("이동비용과 학습 loss")
    cost_axis.set_xlabel("Episode")
    cost_axis.set_ylabel("값")
    cost_axis.grid(alpha=0.25)
    cost_axis.legend()
    cost_axis.text(
        0.01,
        0.86,
        f"이동은 적을수록 좋음 | 최근 100 episode 평균: {recent_movement:.2f}",
        transform=cost_axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#d0d7de"},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_action_distribution_chart(
    action_counts: dict[int, int],
    station_names: list[str],
    output_path: Path = ACTION_DISTRIBUTION_CHART_PATH,
) -> Path:
    """DQN이 평가 중 선택한 대여소 action 분포 그래프를 저장한다."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib이 설치되어 있지 않아 그래프를 저장하지 못했습니다. "
            "`pip install -e .` 또는 `pip install matplotlib` 후 다시 실행하세요."
        ) from exc

    _configure_korean_font(plt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [station_names[action] for action in action_counts]
    values = [action_counts[action] for action in action_counts]
    total = max(1, sum(values))
    colors = ["#2f80ed", "#27ae60", "#f2994a", "#9b51e0", "#eb5757"]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, color=colors[: len(labels)])
    axis.set_title("DQN Action Distribution", fontsize=14, fontweight="bold")
    axis.set_ylabel("선택 횟수")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        axis.text(index, value, f"{value}회\n{value / total:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_multiseed_summary_chart(
    rows: list[dict[str, float]],
    output_path: Path = DQN_MULTI_SEED_CHART_PATH,
) -> Path:
    """seed별 DQN 평가 reward와 unmet demand 분포를 그래프로 저장한다."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib이 설치되어 있지 않아 그래프를 저장하지 못했습니다. "
            "`pip install -e .` 또는 `pip install matplotlib` 후 다시 실행하세요."
        ) from exc

    _configure_korean_font(plt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = [str(int(row["seed"])) for row in rows]
    rewards = [row["avg_reward"] for row in rows]
    unmet_values = [row["avg_unmet_demand"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("DQN Multi-seed 안정성 확인", fontsize=15, fontweight="bold")
    _draw_bar(axes[0], seeds, rewards, "Seed별 평균 보상", higher_is_better=True)
    _draw_bar(axes[1], seeds, unmet_values, "Seed별 미충족 수요", higher_is_better=False)
    axes[0].set_xlabel("seed")
    axes[1].set_xlabel("seed")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_algorithm_comparison_chart(
    rows: list[dict[str, float | str]],
    output_path: Path = ALGORITHM_COMPARISON_CHART_PATH,
) -> Path:
    """알고리즘별 reward, unmet, rejected return 비교 그래프를 저장한다."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib이 설치되어 있지 않아 그래프를 저장하지 못했습니다. "
            "`pip install -e .` 또는 `pip install matplotlib` 후 다시 실행하세요."
        ) from exc

    _configure_korean_font(plt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["algorithm"]) for row in rows]
    rewards = [float(row["avg_reward"]) for row in rows]
    unmet_values = [float(row["avg_unmet_demand"]) for row in rows]
    rejected_values = [float(row["avg_rejected_returns"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle("알고리즘별 성능 비교", fontsize=15, fontweight="bold")
    _draw_bar(axes[0], labels, rewards, "평균 보상", higher_is_better=True)
    _draw_bar(axes[1], labels, unmet_values, "미충족 수요", higher_is_better=False)
    _draw_bar(axes[2], labels, rejected_values, "반납 실패", higher_is_better=False)
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _moving_average(values: list[float], window_size: int = 20) -> list[float]:
    """학습 곡선을 읽기 쉽도록 이동평균 값을 계산한다."""
    averages = []
    for index in range(len(values)):
        window = values[max(0, index - window_size + 1): index + 1]
        averages.append(sum(window) / len(window))
    return averages


def _metric_values(metrics: list[dict[str, float]], key: str) -> list[float]:
    """학습 metric list에서 특정 key 값을 float list로 추출한다."""
    return [float(metric.get(key, 0.0)) for metric in metrics]


def _best_so_far(values: list[float]) -> list[float]:
    """각 episode까지 관측된 이동평균 reward의 최고값을 누적 계산한다."""
    best_values = []
    best_value = float("-inf")
    for value in values:
        best_value = max(best_value, value)
        best_values.append(best_value)
    return best_values


def _recent_average(values: list[float], window_size: int = 100) -> float:
    """마지막 window_size개 episode의 평균값을 반환한다."""
    if not values:
        return 0.0
    window = values[-window_size:]
    return sum(window) / len(window)


def _configure_korean_font(plt: object) -> None:
    """그래프의 한국어 label이 깨지지 않도록 한글 폰트 후보를 설정한다."""
    from matplotlib import font_manager

    preferred_fonts = ("AppleGothic", "Malgun Gothic", "NanumGothic")
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    korean_font = next(
        (font_name for font_name in preferred_fonts if font_name in installed_fonts),
        "DejaVu Sans",
    )
    plt.rcParams["font.family"] = korean_font
    plt.rcParams["axes.unicode_minus"] = False


def _draw_bar(
    axis: object,
    labels: list[str],
    values: list[float],
    title: str,
    higher_is_better: bool,
) -> None:
    """하나의 metric에 대한 막대 그래프를 그리고 최고 policy를 강조한다."""
    best_value = max(values) if higher_is_better else min(values)
    colors = ["#2f80ed" if value == best_value else "#a9b4c2" for value in values]
    axis.bar(labels, values, color=colors)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.2f}", ha="center", va=_label_vertical_alignment(value))


def _label_vertical_alignment(value: float) -> str:
    """막대 값의 부호에 따라 숫자 label 위치를 보기 좋게 정한다."""
    return "top" if value < 0 else "bottom"
