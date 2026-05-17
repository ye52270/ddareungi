"""실제 데이터 profile JSON을 ToyDdareungiEnv config로 변환한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ddareungi_rl.envs.toy_ddareungi_env import ToyDdareungiConfig


def hourly_ranges_to_pattern(
    ranges_by_hour: dict[str, list[list[int]]],
) -> dict[range, tuple[tuple[int, int], ...]]:
    """JSON의 hour별 범위를 ToyDdareungiConfig pattern 구조로 변환한다."""
    pattern: dict[range, tuple[tuple[int, int], ...]] = {}
    for hour_text, station_ranges in sorted(
        ranges_by_hour.items(),
        key=lambda item: int(item[0]),
    ):
        hour = int(hour_text)
        pattern[range(hour, hour + 1)] = tuple(
            (int(low), int(high)) for low, high in station_ranges
        )
    return pattern


def load_profile_payload(profile_path: Path) -> dict[str, Any]:
    """UTF-8 profile JSON 파일을 dict로 읽는다."""
    return json.loads(profile_path.read_text(encoding="utf-8"))


def load_profile_config(
    profile_path: Path,
    base_config: ToyDdareungiConfig | None = None,
) -> ToyDdareungiConfig:
    """profile JSON의 demand/return range를 ToyDdareungiConfig로 변환한다."""
    payload = load_profile_payload(profile_path)
    stations = payload["stations"]
    config = base_config or ToyDdareungiConfig()
    return ToyDdareungiConfig(
        station_count=len(stations),
        station_capacity=config.station_capacity,
        truck_capacity=config.truck_capacity,
        target_stock=config.target_stock,
        episode_steps=config.episode_steps,
        unmet_demand_penalty=config.unmet_demand_penalty,
        movement_cost_value=config.movement_cost_value,
        initial_stock_min=config.initial_stock_min,
        initial_stock_max=config.initial_stock_max,
        initial_truck_bikes=config.initial_truck_bikes,
        initial_truck_location=min(config.initial_truck_location, len(stations) - 1),
        demand_pattern=hourly_ranges_to_pattern(payload["demand_ranges_by_hour"]),
        return_pattern=hourly_ranges_to_pattern(payload["return_ranges_by_hour"]),
    )
