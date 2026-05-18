"""실험 결과를 한 화면에서 보는 정적 HTML dashboard 생성기."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


DASHBOARD_PATH = Path("outputs/reports/experiment_dashboard.html")


def save_experiment_dashboard(
    *,
    output_path: Path = DASHBOARD_PATH,
    report_dir: Path = Path("outputs/reports"),
    figure_dir: Path = Path("outputs/figures"),
    algorithm_name: str = "dqn",
    comparison_path: Path | None = None,
    training_history_path: Path | None = None,
    action_distribution_path: Path | None = None,
    evaluation_episodes_path: Path | None = None,
) -> Path:
    """저장된 report CSV/JSON/PNG를 읽어 직관적인 HTML dashboard를 만든다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = _read_json(report_dir / "experiment_config.json")
    comparison_rows = _read_csv(comparison_path or report_dir / "baseline_vs_dqn.csv")
    algorithm_rows = _read_csv(report_dir / "algorithm_comparison.csv")
    action_distribution_csv = action_distribution_path or _preferred_report_path(
        report_dir,
        f"{algorithm_name}_action_distribution.csv",
        "action_distribution.csv",
    )
    evaluation_episodes_csv = evaluation_episodes_path or _preferred_report_path(
        report_dir,
        f"{algorithm_name}_evaluation_episodes.csv",
        "dqn_evaluation_episodes.csv",
    )
    action_rows = _read_csv(action_distribution_csv)
    history_rows = _read_csv(training_history_path or report_dir / "dqn_training_history.csv")
    episode_rows = _read_csv(evaluation_episodes_csv)
    best_policy = _best_policy(comparison_rows)
    algorithm_row = _find_policy(comparison_rows, algorithm_name)
    low_stock_row = _find_policy(comparison_rows, "low-stock")
    interpretation = _interpret_result(algorithm_row, low_stock_row, best_policy, algorithm_name)

    output_path.write_text(
        _render_html(
            algorithm_name=algorithm_name,
            config=config,
            comparison_rows=comparison_rows,
            algorithm_rows=algorithm_rows,
            action_rows=action_rows,
            history_rows=history_rows,
            episode_rows=episode_rows,
            best_policy=best_policy,
            algorithm_row=algorithm_row,
            interpretation=interpretation,
            figure_dir=figure_dir,
            action_distribution_figure_path=_preferred_figure_path(
                figure_dir,
                f"{algorithm_name}_action_distribution.png",
                "action_distribution.png",
            ),
            step_trace_path=_preferred_report_path(
                report_dir,
                f"{algorithm_name}_step_trace.csv",
                "dqn_step_trace.csv",
            ),
        ),
        encoding="utf-8",
    )
    return output_path


def _render_html(
    *,
    algorithm_name: str,
    config: dict[str, Any],
    comparison_rows: list[dict[str, str]],
    algorithm_rows: list[dict[str, str]],
    action_rows: list[dict[str, str]],
    history_rows: list[dict[str, str]],
    episode_rows: list[dict[str, str]],
    best_policy: dict[str, str],
    algorithm_row: dict[str, str],
    interpretation: str,
    figure_dir: Path,
    action_distribution_figure_path: Path,
    step_trace_path: Path,
) -> str:
    """dashboard HTML 문자열을 만든다."""
    environment = config.get("environment", {})
    dqn_config = config.get("dqn_config", {})
    mdp = config.get("mdp", {})
    recent_training = _recent_training_summary(history_rows)
    display_name = _display_algorithm(algorithm_name)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ddareungi RL Experiment Dashboard</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #17212b;
      --muted: #667085;
      --line: #d9e2ec;
      --blue: #2f80ed;
      --green: #27ae60;
      --orange: #f2994a;
      --red: #eb5757;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 36px 24px 56px; }}
    header {{ margin-bottom: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    p {{ margin: 0 0 12px; }}
    .muted {{ color: var(--muted); }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .badge {{
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }}
    .grid {{ display: grid; gap: 16px; }}
    .cards {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 20px 0; }}
    .two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 18px;
      box-shadow: 0 8px 20px rgba(16, 24, 40, 0.05);
    }}
    .metric {{ font-size: 30px; font-weight: 800; margin-top: 8px; }}
    .metric.good {{ color: var(--green); }}
    .metric.bad {{ color: var(--red); }}
    .metric.blue {{ color: var(--blue); }}
    .section {{ margin-top: 22px; }}
    .callout {{
      border-left: 5px solid var(--blue);
      background: #eef5ff;
      padding: 16px 18px;
      border-radius: 8px;
      margin: 18px 0;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: #f8fafc; }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .figure {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 8px;
    }}
    .figure img {{ width: 100%; display: block; border-radius: 6px; }}
    code {{
      background: #eef2f7;
      border: 1px solid #dde5ef;
      border-radius: 5px;
      padding: 1px 5px;
      font-size: 0.94em;
    }}
    @media (max-width: 900px) {{
      .cards, .two {{ grid-template-columns: 1fr; }}
      main {{ padding: 24px 14px 40px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>따릉이 RL Experiment Dashboard</h1>
    <p class="muted">{_escape(display_name)} 학습, baseline 평가, 행동 분석 결과를 한 화면에서 확인한다.</p>
    <div class="badge-row">
      <span class="badge">Profile: {_escape(environment.get("profile_path", "-"))}</span>
      <span class="badge">Stations: {_escape(", ".join(environment.get("station_names", [])))}</span>
      <span class="badge">Train Episodes: {_escape(dqn_config.get("episodes", "-"))}</span>
      <span class="badge">Daily Dates: {_escape(environment.get("daily_profile_dates", "-"))}</span>
    </div>
  </header>

  <section class="grid cards">
    {_metric_card("Best Policy", best_policy.get("policy", "-"), "blue")}
    {_metric_card(f"{display_name} Avg Reward", _fmt(algorithm_row.get("avg_reward")), "good")}
    {_metric_card(f"{display_name} Unmet", _fmt(algorithm_row.get("avg_unmet_demand")), "")}
    {_metric_card(f"{display_name} Rejected", _fmt(algorithm_row.get("avg_rejected_returns")), "")}
  </section>

  <section class="callout">
    <strong>해석 요약</strong>
    <p>{_escape(interpretation)}</p>
  </section>

  <section class="section grid two">
    <div class="card">
      <h2>MDP 설정</h2>
      <table>
        <tr><th>State</th><td>{_escape(", ".join(mdp.get("state", [])))}</td></tr>
        <tr><th>Action</th><td>{_escape(mdp.get("action", "-"))}</td></tr>
        <tr><th>Reward</th><td><code>{_escape(mdp.get("reward", "-"))}</code></td></tr>
        <tr><th>Environment</th><td>{_escape(environment.get("station_count", "-"))} stations, {_escape(environment.get("episode_steps", "-"))} steps/day</td></tr>
      </table>
    </div>
    <div class="card">
      <h2>최근 학습 상태</h2>
      <table>
        <tr><th>Last Episode</th><td class="num">{_escape(recent_training["episode"])}</td></tr>
        <tr><th>Last Reward</th><td class="num">{_escape(recent_training["reward"])}</td></tr>
        <tr><th>Last Unmet</th><td class="num">{_escape(recent_training["unmet_demand"])}</td></tr>
        <tr><th>Last Epsilon</th><td class="num">{_escape(recent_training["epsilon"])}</td></tr>
      </table>
    </div>
  </section>

  <section class="section card">
    <h2>Baseline vs {_escape(display_name)} 성능표</h2>
    {_table(comparison_rows, ["policy", "avg_reward", "avg_unmet_demand", "avg_rejected_returns", "avg_movement_cost", "avg_service_rate", "same_location_rate"])}
  </section>

  {_algorithm_comparison_section(algorithm_rows, figure_dir / "algorithm_comparison.png")}

  <section class="section grid two">
    {_figure(f"Baseline vs {display_name} 비교", _comparison_figure_path(figure_dir, algorithm_name))}
    {_figure(f"{display_name} 학습 곡선", _training_figure_path(figure_dir, algorithm_name))}
    {_figure(f"{display_name} Action Distribution", action_distribution_figure_path)}
    {_figure("Baseline 정책 비교", figure_dir / "baseline_comparison.png")}
  </section>

  <section class="section grid two">
    <div class="card">
      <h2>Action Distribution</h2>
      {_table(action_rows, ["station_name", "count", "ratio"])}
    </div>
    <div class="card">
      <h2>{_escape(display_name)} 평가 Episode 미리보기</h2>
      {_table(episode_rows[:10], ["episode", "date", "reward", "unmet_demand", "rejected_returns", "service_rate"])}
      <p class="muted">전체 trace는 <code>{_escape(step_trace_path)}</code>에서 확인한다.</p>
    </div>
  </section>
</main>
</body>
</html>
"""


def _preferred_report_path(report_dir: Path, preferred_name: str, legacy_name: str) -> Path:
    """알고리즘별 report가 있으면 우선 사용하고, 없으면 기존 파일명으로 fallback한다."""
    preferred_path = report_dir / preferred_name
    if preferred_path.exists():
        return preferred_path
    return report_dir / legacy_name


def _preferred_figure_path(figure_dir: Path, preferred_name: str, legacy_name: str) -> Path:
    """알고리즘별 figure가 있으면 우선 사용하고, 없으면 기존 파일명으로 fallback한다."""
    preferred_path = figure_dir / preferred_name
    if preferred_path.exists():
        return preferred_path
    return figure_dir / legacy_name


def _algorithm_comparison_section(rows: list[dict[str, str]], image_path: Path) -> str:
    """저장된 알고리즘별 결과가 있으면 전체 비교 section을 만든다."""
    if not rows:
        return ""

    figure_html = ""
    if image_path.exists():
        relative_path = Path("..") / "figures" / image_path.name
        figure_html = (
            '<div class="figure" style="margin-top:14px;">'
            f'<img src="{_escape(relative_path)}" alt="전체 알고리즘 비교">'
            "</div>"
        )

    return f"""
  <section class="section card">
    <h2>전체 알고리즘 비교</h2>
    <p class="muted">Baseline, DQN, Double DQN, Dueling DQN을 같은 평가 episode 기준으로 비교한다.</p>
    {_table(rows, ["algorithm", "avg_reward", "avg_unmet_demand", "avg_rejected_returns", "avg_movement_cost", "avg_service_rate", "same_location_rate"])}
    {figure_html}
  </section>
"""


def _metric_card(label: str, value: object, color_class: str) -> str:
    """상단 summary card HTML을 만든다."""
    color = f" {color_class}" if color_class else ""
    return f"""
    <div class="card">
      <h3>{_escape(label)}</h3>
      <div class="metric{color}">{_escape(value)}</div>
    </div>
    """


def _display_algorithm(algorithm_name: str) -> str:
    """내부 알고리즘 id를 화면 표시용 이름으로 바꾼다."""
    return {
        "dqn": "DQN",
        "double_dqn": "Double DQN",
        "dueling_dqn": "Dueling DQN",
    }.get(algorithm_name, algorithm_name)


def _figure(title: str, image_path: Path) -> str:
    """이미지가 있으면 figure block을 만들고, 없으면 안내문을 만든다."""
    if not image_path.exists():
        return f"""
        <div class="card">
          <h2>{_escape(title)}</h2>
          <p class="muted">이미지 파일이 아직 없습니다: <code>{_escape(image_path)}</code></p>
        </div>
        """
    relative_path = Path("..") / "figures" / image_path.name
    return f"""
    <div class="card">
      <h2>{_escape(title)}</h2>
      <div class="figure"><img src="{_escape(relative_path)}" alt="{_escape(title)}"></div>
    </div>
    """


def _table(rows: list[dict[str, str]], columns: list[str]) -> str:
    """선택 column만 간단한 HTML table로 렌더링한다."""
    if not rows:
        return '<p class="muted">표시할 데이터가 없습니다.</p>'
    header = "".join(f"<th>{_escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = _fmt(row.get(column, ""))
            css_class = ' class="num"' if _is_number(value) else ""
            cells.append(f"<td{css_class}>{_escape(value)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _best_policy(rows: list[dict[str, str]]) -> dict[str, str]:
    """avg_reward가 가장 높은 policy row를 반환한다."""
    if not rows:
        return {}
    return max(rows, key=lambda row: _to_float(row.get("avg_reward")))


def _find_policy(rows: list[dict[str, str]], policy_name: str) -> dict[str, str]:
    """policy 이름으로 결과 row를 찾는다."""
    return next((row for row in rows if row.get("policy") == policy_name), {})


def _comparison_figure_path(figure_dir: Path, algorithm_name: str) -> Path:
    """알고리즘 이름에 맞는 baseline 비교 그래프 경로를 반환한다."""
    if algorithm_name == "dqn":
        return figure_dir / "dqn_vs_baseline_comparison.png"
    return figure_dir / f"{algorithm_name}_vs_baseline_comparison.png"


def _training_figure_path(figure_dir: Path, algorithm_name: str) -> Path:
    """알고리즘 이름에 맞는 학습 곡선 경로를 반환한다."""
    if algorithm_name == "dqn":
        return figure_dir / "dqn_training_curve.png"
    return figure_dir / f"{algorithm_name}_training_curve.png"


def _interpret_result(
    algorithm_row: dict[str, str],
    low_stock_row: dict[str, str],
    best_policy: dict[str, str],
    algorithm_name: str,
) -> str:
    """선택 알고리즘과 low-stock baseline을 비교한 짧은 해석 문장을 만든다."""
    if not algorithm_row:
        return f"{algorithm_name} 평가 결과가 아직 없습니다."
    if not low_stock_row:
        return f"{algorithm_name} 결과는 저장되었지만 low-stock baseline과 비교할 수 없습니다."

    reward_gap = _gap(algorithm_row, low_stock_row, "avg_reward")
    unmet_gap = _gap(algorithm_row, low_stock_row, "avg_unmet_demand")
    rejected_gap = _gap(algorithm_row, low_stock_row, "avg_rejected_returns")
    best_text = f"현재 avg reward 기준 최고 정책은 {best_policy.get('policy', '-')}이다."
    return (
        f"{best_text} {algorithm_name}은 low-stock 대비 reward가 {reward_gap:+.2f}, "
        f"미충족 수요가 {unmet_gap:+.2f}, 반납 실패가 {rejected_gap:+.2f} 차이난다. "
        "reward는 높을수록 좋고, 미충족 수요와 반납 실패는 낮을수록 좋다."
    )


def _recent_training_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    """마지막 학습 episode row에서 dashboard summary 값을 뽑는다."""
    if not rows:
        return {"episode": "-", "reward": "-", "unmet_demand": "-", "epsilon": "-"}
    row = rows[-1]
    return {
        "episode": _fmt(row.get("episode")),
        "reward": _fmt(row.get("reward")),
        "unmet_demand": _fmt(row.get("unmet_demand")),
        "epsilon": _fmt(row.get("epsilon")),
    }


def _gap(row: dict[str, str], baseline_row: dict[str, str], metric: str) -> float:
    """두 row의 metric 차이를 안전하게 계산한다."""
    return (_to_float(row.get(metric)) or 0.0) - (_to_float(baseline_row.get(metric)) or 0.0)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """CSV 파일을 dict row 목록으로 읽는다."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, Any]:
    """JSON 파일을 dict로 읽는다."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    """dashboard에서 읽기 쉬운 숫자 문자열을 만든다."""
    number = _to_float(value)
    if number is None:
        return str(value or "-")
    if abs(number) >= 100:
        return f"{number:.0f}"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _to_float(value: object) -> float | None:
    """값을 float로 변환하고 실패하면 None을 반환한다."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _is_number(value: str) -> bool:
    """문자열이 숫자로 해석 가능한지 확인한다."""
    return _to_float(value) is not None


def _escape(value: object) -> str:
    """HTML 특수문자를 escape한다."""
    return html.escape(str(value))
