"""논문식 실험 결과를 CSV/Markdown 파일로 저장하는 도구."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from ddareungi_rl.dqn import DQNConfig
from ddareungi_rl.env import DdareungiEnv


REPORT_DIR = Path("outputs/reports")
EXPERIMENT_CONFIG_PATH = REPORT_DIR / "experiment_config.json"
MDP_SUMMARY_PATH = REPORT_DIR / "mdp_summary.md"
BASELINE_VS_DQN_PATH = REPORT_DIR / "baseline_vs_dqn.csv"
DQN_TRAINING_HISTORY_PATH = REPORT_DIR / "dqn_training_history.csv"
ACTION_DISTRIBUTION_PATH = REPORT_DIR / "action_distribution.csv"
DQN_EVALUATION_EPISODES_PATH = REPORT_DIR / "dqn_evaluation_episodes.csv"
DQN_STEP_TRACE_PATH = REPORT_DIR / "dqn_step_trace.csv"
DQN_MULTI_SEED_RUNS_PATH = REPORT_DIR / "dqn_multiseed_runs.csv"
DQN_MULTI_SEED_SUMMARY_PATH = REPORT_DIR / "dqn_multiseed_summary.csv"
ALGORITHM_COMPARISON_PATH = REPORT_DIR / "algorithm_comparison.csv"


def save_experiment_config(
    *,
    env: DdareungiEnv,
    config: DQNConfig,
    profile_path: Path,
    output_path: Path = EXPERIMENT_CONFIG_PATH,
) -> Path:
    """실험 설정을 report에서 재사용하기 쉬운 JSON으로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": {
            "station_names": list(env.config.station_names),
            "station_count": env.config.station_count,
            "episode_steps": env.config.episode_steps,
            "station_capacity": env.config.station_capacity,
            "truck_capacity": env.config.truck_capacity,
            "target_stock": env.config.target_stock,
            "profile_path": str(profile_path),
            "daily_profile_dates": len(env.config.daily_dates),
        },
        "mdp": {
            "state": [
                "station_bikes",
                "expected_demand",
                "truck_location",
                "truck_bikes",
                "time_step",
            ],
            "action": "next station index to visit",
            "reward": (
                f"-{env.config.unmet_penalty} * unmet_demand "
                f"- {env.config.full_penalty} * rejected_returns "
                f"- {env.config.move_cost} * movement_cost"
            ),
        },
        "dqn_config": asdict(config),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def save_mdp_summary(env: DdareungiEnv, output_path: Path = MDP_SUMMARY_PATH) -> Path:
    """State/Action/Reward/Environment 요약을 Markdown 표로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# MDP Summary

| Item | Definition |
|---|---|
| State | station_bikes, expected_demand, truck_location, truck_bikes, time_step |
| Action | next station index to visit |
| Reward | -{env.config.unmet_penalty} * unmet_demand - {env.config.full_penalty} * rejected_returns - {env.config.move_cost} * movement_cost |
| Environment | {env.config.station_count} stations, 1 truck, {env.config.episode_steps} steps per day |
| Goal | reduce stockout and rejected-return events |

## Stations

{_station_list(env)}
"""
    output_path.write_text(text, encoding="utf-8")
    return output_path


def save_baseline_vs_dqn_csv(
    results: dict[str, dict[str, float]],
    output_path: Path = BASELINE_VS_DQN_PATH,
) -> Path:
    """baseline과 DQN 성능 비교표를 CSV로 저장한다."""
    fieldnames = [
        "policy",
        "avg_reward",
        "avg_unmet_demand",
        "avg_rejected_returns",
        "avg_movement_cost",
        "avg_service_rate",
        "same_location_rate",
    ]
    rows = [
        {"policy": policy_name, **_metric_defaults(metrics)}
        for policy_name, metrics in results.items()
    ]
    _write_csv(output_path, fieldnames, rows)
    return output_path


def save_training_history_csv(
    metrics: list[dict[str, float]],
    output_path: Path = DQN_TRAINING_HISTORY_PATH,
) -> Path:
    """DQN episode별 학습 history를 CSV로 저장한다."""
    fieldnames = [
        "episode",
        "reward",
        "unmet_demand",
        "rejected_returns",
        "movement_cost",
        "epsilon",
        "loss",
    ]
    _write_csv(output_path, fieldnames, metrics)
    return output_path


def save_policy_trace_reports(
    *,
    policy_name: str,
    station_names: list[str],
    evaluation_report: dict[str, Any],
    report_dir: Path = REPORT_DIR,
) -> dict[str, Path]:
    """DQN 평가의 action 분포, episode 결과, step trace를 CSV로 저장한다."""
    report_dir.mkdir(parents=True, exist_ok=True)
    action_rows = [
        {
            "policy": policy_name,
            "action": action,
            "station_name": station_names[action],
            "count": count,
            "ratio": count / max(1, sum(evaluation_report["action_counts"].values())),
        }
        for action, count in evaluation_report["action_counts"].items()
    ]
    _write_csv(
        ACTION_DISTRIBUTION_PATH,
        ["policy", "action", "station_name", "count", "ratio"],
        action_rows,
    )
    _write_csv(
        DQN_EVALUATION_EPISODES_PATH,
        [
            "policy",
            "episode",
            "date",
            "reward",
            "served_demand",
            "unmet_demand",
            "rejected_returns",
            "movement_cost",
            "service_rate",
            "same_location_steps",
        ],
        evaluation_report["episodes"],
    )
    _write_csv(
        DQN_STEP_TRACE_PATH,
        [
            "policy",
            "episode",
            "date",
            "time_step",
            "action",
            "previous_truck_location",
            "truck_location",
            "truck_bikes",
            "reward",
            "served_demand",
            "unmet_demand",
            "rejected_returns",
            "movement_cost",
            "moved_bikes",
            "station_bikes",
            "demand",
            "returns",
        ],
        evaluation_report["steps"],
    )
    return {
        "action_distribution": ACTION_DISTRIBUTION_PATH,
        "evaluation_episodes": DQN_EVALUATION_EPISODES_PATH,
        "step_trace": DQN_STEP_TRACE_PATH,
    }


def save_multiseed_reports(
    rows: list[dict[str, float]],
    *,
    runs_path: Path = DQN_MULTI_SEED_RUNS_PATH,
    summary_path: Path = DQN_MULTI_SEED_SUMMARY_PATH,
) -> dict[str, Path]:
    """seed별 DQN 평가 결과와 평균/표준편차 summary를 CSV로 저장한다."""
    run_fields = [
        "seed",
        "avg_reward",
        "avg_unmet_demand",
        "avg_rejected_returns",
        "avg_movement_cost",
        "avg_service_rate",
        "same_location_rate",
    ]
    _write_csv(runs_path, run_fields, rows)
    summary_rows = _multiseed_summary_rows(rows, run_fields[1:])
    _write_csv(summary_path, ["metric", "mean", "std", "min", "max"], summary_rows)
    return {"runs": runs_path, "summary": summary_path}


def save_algorithm_comparison_from_reports(
    *,
    report_dir: Path = REPORT_DIR,
    output_path: Path = ALGORITHM_COMPARISON_PATH,
) -> Path:
    """저장된 DQN 계열 결과 CSV들을 모아 알고리즘 비교표를 만든다."""
    algorithm_files = {
        "dqn": report_dir / "baseline_vs_dqn.csv",
        "double_dqn": report_dir / "double_dqn_vs_baseline.csv",
        "dueling_dqn": report_dir / "dueling_dqn_vs_baseline.csv",
    }
    rows = []
    baseline_row = None
    for algorithm_name, path in algorithm_files.items():
        if not path.exists():
            continue
        file_rows = _read_csv_rows(path)
        if baseline_row is None:
            baseline_row = _find_row(file_rows, "low-stock")
        algorithm_row = _find_row(file_rows, algorithm_name)
        if algorithm_row:
            rows.append({"algorithm": algorithm_name, **_metric_defaults_str(algorithm_row)})
    if baseline_row:
        rows.insert(0, {"algorithm": "low-stock", **_metric_defaults_str(baseline_row)})
    _write_csv(
        output_path,
        [
            "algorithm",
            "avg_reward",
            "avg_unmet_demand",
            "avg_rejected_returns",
            "avg_movement_cost",
            "avg_service_rate",
            "same_location_rate",
        ],
        rows,
    )
    return output_path


def _metric_defaults(metrics: dict[str, float]) -> dict[str, float]:
    """이전 결과 dict에 없는 metric은 0으로 채워 CSV schema를 안정화한다."""
    return {
        "avg_reward": metrics.get("avg_reward", 0.0),
        "avg_unmet_demand": metrics.get("avg_unmet_demand", 0.0),
        "avg_rejected_returns": metrics.get("avg_rejected_returns", 0.0),
        "avg_movement_cost": metrics.get("avg_movement_cost", 0.0),
        "avg_service_rate": metrics.get("avg_service_rate", 0.0),
        "same_location_rate": metrics.get("same_location_rate", 0.0),
    }


def _multiseed_summary_rows(
    rows: list[dict[str, float]],
    metrics: list[str],
) -> list[dict[str, float | str]]:
    """seed별 결과에서 metric별 평균/표준편차/min/max row를 만든다."""
    return [
        {
            "metric": metric,
            "mean": _mean([row[metric] for row in rows]),
            "std": _std([row[metric] for row in rows]),
            "min": min(row[metric] for row in rows),
            "max": max(row[metric] for row in rows),
        }
        for metric in metrics
        if rows
    ]


def _mean(values: list[float]) -> float:
    """숫자 목록의 평균을 반환한다."""
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    """표본 표준편차가 아닌 간단한 population std를 반환한다."""
    if not values:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _station_list(env: DdareungiEnv) -> str:
    """Markdown에 넣을 대여소 목록을 만든다."""
    return "\n".join(f"- {index}: {name}" for index, name in enumerate(env.config.station_names))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """dict row 목록을 UTF-8 CSV로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """CSV 파일을 dict row 목록으로 읽는다."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _find_row(rows: list[dict[str, str]], name: str) -> dict[str, str] | None:
    """policy 이름과 일치하는 row를 찾는다."""
    return next((row for row in rows if row.get("policy") == name), None)


def _metric_defaults_str(metrics: dict[str, str]) -> dict[str, float]:
    """CSV 문자열 metric을 float dict로 변환한다."""
    return {
        "avg_reward": float(metrics.get("avg_reward", 0.0)),
        "avg_unmet_demand": float(metrics.get("avg_unmet_demand", 0.0)),
        "avg_rejected_returns": float(metrics.get("avg_rejected_returns", 0.0)),
        "avg_movement_cost": float(metrics.get("avg_movement_cost", 0.0)),
        "avg_service_rate": float(metrics.get("avg_service_rate", 0.0)),
        "same_location_rate": float(metrics.get("same_location_rate", 0.0)),
    }
