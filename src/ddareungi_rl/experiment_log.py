"""DQN 실험 parameter와 결과를 report용 로그로 저장한다."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from ddareungi_rl.dqn import DQNConfig
from ddareungi_rl.env import DdareungiEnv


DQN_EXPERIMENT_LOG_PATH = Path("outputs/experiments/dqn_runs.jsonl")


def append_dqn_experiment_log(
    *,
    config: DQNConfig,
    env: DdareungiEnv,
    eval_episodes: int,
    eval_result: dict[str, float],
    last_training_metric: dict[str, float],
    model_path: Path,
    curve_path: Path | None,
    profile_path: Path,
    log_path: Path = DQN_EXPERIMENT_LOG_PATH,
) -> Path:
    """DQN 학습 설정과 평가 결과를 JSONL 파일에 한 줄 추가한다."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "DQN",
        "profile_path": str(profile_path),
        "model_path": str(model_path),
        "training_curve_path": str(curve_path) if curve_path else "",
        "state_design": "station_stock + expected_demand + truck_location + truck_bikes + time",
        "observation_size": int(env.observation_space.shape[0]),
        "action_size": int(env.action_space.n),
        "train_config": asdict(config),
        "eval_episodes": eval_episodes,
        "eval_result": eval_result,
        "last_training_metric": last_training_metric,
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return log_path


def read_experiment_log(log_path: Path = DQN_EXPERIMENT_LOG_PATH) -> list[dict[str, Any]]:
    """저장된 DQN 실험 로그를 report 작성용 list로 읽는다."""
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
