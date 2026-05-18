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
    # YAML은 실험 조건을, JSON은 시간대별 수요/반납 샘플 범위를 담당한다.
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sample_path = _resolve_path(config_path, payload["simulation"]["sample_data"])
    sample_payload = json.loads(sample_path.read_text(encoding="utf-8"))

    # EnvConfig 생성 시 __post_init__ 검증이 함께 실행된다.
    return EnvConfig(
        station_names=tuple(payload["station"]["names"]),
        station_capacity=int(payload["station"]["capacity"]),
        initial_stock_min=int(payload["station"]["initial_stock_min"]),
        initial_stock_max=int(payload["station"]["initial_stock_max"]),
        truck_capacity=int(payload["truck"]["capacity"]),
        target_stock=int(payload["simulation"]["target_stock"]),
        episode_steps=int(payload["simulation"]["episode_steps"]),
        unmet_penalty=int(payload["reward"]["unmet_penalty"]),
        full_penalty=int(payload["reward"]["full_penalty"]),
        move_cost=int(payload["reward"]["move_cost"]),
        initial_truck_bikes=int(payload["truck"]["initial_bikes"]),
        traffic_enabled=bool(payload.get("traffic", {}).get("enabled", False)),
        traffic_factors=_parse_traffic_factors(
            payload.get("traffic", {}),
            episode_steps=int(payload["simulation"]["episode_steps"]),
        ),
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


def _parse_traffic_factors(raw: dict[str, object], episode_steps: int) -> tuple[float, ...]:
    """YAML의 시간대별 traffic factor 범위를 episode_steps 길이 tuple로 펼친다."""
    if not raw.get("enabled", False):
        return ()
    factors = [None for _ in range(episode_steps)]
    raw_factors = raw.get("factors", {})
    if not isinstance(raw_factors, dict):
        raise ValueError("traffic.factors must be a mapping such as '8-8': 1.5")

    for hour_range, factor in raw_factors.items():
        start, end = _parse_hour_range(str(hour_range), episode_steps)
        factor_value = float(factor)
        if factor_value <= 0:
            raise ValueError("traffic factor must be positive")
        for hour in range(start, end + 1):
            if factors[hour] is not None:
                raise ValueError(f"traffic factor for hour {hour} is duplicated")
            factors[hour] = factor_value

    missing_hours = [hour for hour, factor in enumerate(factors) if factor is None]
    if missing_hours:
        raise ValueError(f"traffic.factors missing hours: {missing_hours}")
    return tuple(float(factor) for factor in factors)


def _parse_hour_range(raw_range: str, episode_steps: int) -> tuple[int, int]:
    """'8-8' 또는 '0-5' 형식의 hour 범위를 시작/끝 hour로 변환한다."""
    parts = raw_range.split("-")
    if len(parts) != 2:
        raise ValueError(f"traffic hour range must look like '8-8': {raw_range}")
    start, end = int(parts[0]), int(parts[1])
    if not 0 <= start <= end < episode_steps:
        raise ValueError(
            f"traffic hour range must satisfy 0 <= start <= end < {episode_steps}: "
            f"{raw_range}"
        )
    return start, end
