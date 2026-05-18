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
