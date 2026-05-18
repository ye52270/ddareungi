"""YAML 설정 파일과 샘플 데이터를 EnvConfig로 변환한다."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ddareungi_rl.env import EnvConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_env.yaml"


def load_default_config() -> EnvConfig:
    """프로젝트 기본 YAML 설정을 EnvConfig로 읽는다."""
    return load_env_config(DEFAULT_CONFIG_PATH)


def load_env_config(config_path: Path) -> EnvConfig:
    """환경 YAML 설정과 연결된 sample_data를 EnvConfig로 변환한다."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sample_path = _resolve_path(config_path, payload["simulation"]["sample_data"])
    sample_payload = json.loads(sample_path.read_text(encoding="utf-8"))
    return EnvConfig(
        station_names=tuple(payload["station"]["names"]),
        station_capacity=int(payload["station"]["capacity"]),
        truck_capacity=int(payload["truck"]["capacity"]),
        target_stock=int(payload["simulation"]["target_stock"]),
        episode_steps=int(payload["simulation"]["episode_steps"]),
        unmet_penalty=int(payload["reward"]["unmet_penalty"]),
        full_penalty=int(payload["reward"]["full_penalty"]),
        move_cost=int(payload["reward"]["move_cost"]),
        initial_truck_bikes=int(payload["truck"]["initial_bikes"]),
        demand_ranges=_parse_ranges(sample_payload["demand_ranges"]),
        return_ranges=_parse_ranges(sample_payload["return_ranges"]),
    )


def _resolve_path(config_path: Path, raw_path: str) -> Path:
    """설정 파일 기준 상대 경로 또는 프로젝트 기준 경로를 실제 Path로 변환한다."""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return project_candidate
    return config_path.parent / candidate


def _parse_ranges(raw: dict[str, list[list[int]]]) -> dict[int, tuple[tuple[int, int], ...]]:
    """JSON의 hour별 list 범위를 EnvConfig가 쓰는 tuple 구조로 변환한다."""
    return {
        int(hour): tuple((int(low), int(high)) for low, high in station_ranges)
        for hour, station_ranges in raw.items()
    }
