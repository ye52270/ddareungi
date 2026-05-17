"""테스트에서 재사용하는 작은 real-profile JSON fixture helper."""

from __future__ import annotations

import json
from pathlib import Path


def write_tiny_profile(profile_path: Path) -> Path:
    """3개 station, 24시간짜리 작은 profile JSON을 저장하고 경로를 반환한다."""
    demand_ranges = {str(hour): [[0, 1], [0, 1], [0, 1]] for hour in range(24)}
    return_ranges = {str(hour): [[0, 1], [0, 1], [0, 1]] for hour in range(24)}
    demand_ranges["8"] = [[2, 3], [0, 1], [1, 2]]
    return_ranges["18"] = [[0, 1], [2, 3], [0, 1]]
    payload = {
        "metadata": {
            "source": "test fixture",
            "observed_days": 2,
            "start_date": "2025-12-01",
            "end_date": "2025-12-02",
        },
        "stations": [
            {"id": "ST-2031", "name": "마곡나루역 2번 출구", "lat": 37.566925, "lon": 126.827438},
            {"id": "ST-2033", "name": "LG유플러스 마곡사옥", "lat": 37.561337, "lon": 126.8339},
            {"id": "ST-2049", "name": "마곡수명산 1-2단지", "lat": 37.555309, "lon": 126.829857},
        ],
        "demand_ranges_by_hour": demand_ranges,
        "return_ranges_by_hour": return_ranges,
    }
    profile_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return profile_path
