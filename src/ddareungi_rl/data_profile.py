"""실제 데이터에서 만든 작은 profile JSON을 환경 설정으로 읽는다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ddareungi_rl.env import EnvConfig


def load_profile(path: Path, base: EnvConfig | None = None) -> EnvConfig:
    """profile JSON을 EnvConfig로 변환한다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_config = base or EnvConfig()
    stations = payload["stations"]
    return EnvConfig(
        station_names=tuple(str(station["name"]) for station in stations),
        station_capacity=base_config.station_capacity,
        truck_capacity=base_config.truck_capacity,
        target_stock=base_config.target_stock,
        episode_steps=base_config.episode_steps,
        unmet_penalty=base_config.unmet_penalty,
        full_penalty=base_config.full_penalty,
        move_cost=base_config.move_cost,
        initial_truck_bikes=base_config.initial_truck_bikes,
        demand_ranges=_ranges(payload["demand_ranges_by_hour"]),
        return_ranges=_ranges(payload["return_ranges_by_hour"]),
    )


def _ranges(raw: dict[str, list[list[int]]]) -> dict[int, tuple[tuple[int, int], ...]]:
    """JSON의 문자열 hour key를 EnvConfig가 쓰는 int hour key로 바꾼다."""
    return {
        int(hour): tuple((int(low), int(high)) for low, high in station_ranges)
        for hour, station_ranges in raw.items()
    }


def profile_summary(path: Path) -> dict[str, Any]:
    """profile 파일의 핵심 메타데이터를 반환한다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "stations": [station["name"] for station in payload["stations"]],
        "metadata": payload.get("metadata", {}),
    }
