"""실제 데이터에서 만든 작은 profile JSON을 환경 설정으로 읽는다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ddareungi_rl.env import EnvConfig


def load_profile(path: Path, base: EnvConfig | None = None) -> EnvConfig:
    """profile JSON을 EnvConfig로 변환한다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if base is None:
        from ddareungi_rl.config_loader import load_default_config

        base = load_default_config()
    base_config = base
    stations = payload["stations"]
    if payload.get("profile_kind") == "daily":
        dates = tuple(sorted(payload["daily_demand_counts"]))
        daily_demand_counts = _daily_counts(payload["daily_demand_counts"], dates)
        daily_return_counts = _daily_counts(payload["daily_return_counts"], dates)
        return EnvConfig(
            station_names=tuple(str(station["name"]) for station in stations),
            station_capacity=base_config.station_capacity,
            initial_stock_min=base_config.initial_stock_min,
            initial_stock_max=base_config.initial_stock_max,
            truck_capacity=base_config.truck_capacity,
            target_stock=base_config.target_stock,
            episode_steps=base_config.episode_steps,
            unmet_penalty=base_config.unmet_penalty,
            full_penalty=base_config.full_penalty,
            move_cost=base_config.move_cost,
            initial_truck_bikes=base_config.initial_truck_bikes,
            demand_ranges=_ranges_from_daily_counts(daily_demand_counts),
            return_ranges=_ranges_from_daily_counts(daily_return_counts),
            daily_dates=dates,
            daily_demand_counts=daily_demand_counts,
            daily_return_counts=daily_return_counts,
        )
    return EnvConfig(
        station_names=tuple(str(station["name"]) for station in stations),
        station_capacity=base_config.station_capacity,
        initial_stock_min=base_config.initial_stock_min,
        initial_stock_max=base_config.initial_stock_max,
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


def _daily_counts(
    raw: dict[str, list[list[int]]],
    dates: tuple[str, ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """daily profile JSON을 날짜/시간/대여소 tuple 구조로 바꾼다."""
    return tuple(
        tuple(tuple(int(value) for value in station_counts) for station_counts in raw[date])
        for date in dates
    )


def _ranges_from_daily_counts(
    daily_counts: tuple[tuple[tuple[int, ...], ...], ...],
) -> dict[int, tuple[tuple[int, int], ...]]:
    """daily count에서 시간대별 최소/최대 범위를 계산한다."""
    hour_count = len(daily_counts[0])
    station_count = len(daily_counts[0][0])
    ranges: dict[int, tuple[tuple[int, int], ...]] = {}
    for hour in range(hour_count):
        hour_ranges = []
        for station_id in range(station_count):
            values = [day_counts[hour][station_id] for day_counts in daily_counts]
            hour_ranges.append((min(values), max(values)))
        ranges[hour] = tuple(hour_ranges)
    return ranges


def profile_summary(path: Path) -> dict[str, Any]:
    """profile 파일의 핵심 메타데이터를 반환한다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "stations": [station["name"] for station in payload["stations"]],
        "metadata": payload.get("metadata", {}),
    }
