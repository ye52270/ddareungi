"""공공 따릉이 CSV에서 강화학습용 시간대별 profile을 만든다."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from math import ceil, floor
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 설치 전에도 테스트 가능한 안전장치다.
    tqdm = None


RENT_TIME_COLUMN = "대여일시"
RETURN_TIME_COLUMN = "반납일시"
RENT_STATION_ID_COLUMN = "대여대여소ID"
RETURN_STATION_ID_COLUMN = "반납대여소ID"
RENT_STATION_NAME_COLUMN = "대여 대여소명"
RETURN_STATION_NAME_COLUMN = "반납대여소명"


@dataclass(frozen=True)
class StationCandidate:
    """전처리 대상 후보 대여소의 기본 정보를 담는다."""

    station_id: str
    name: str
    count: int


def build_profile_from_csvs(
    rental_paths: list[Path],
    output_path: Path,
    station_keyword: str = "마곡",
    station_count: int = 3,
    station_ids: tuple[str, ...] = (),
    master_csv: Path | None = None,
    encoding: str = "cp949",
    max_sample_high: int = 5,
    scale: float | None = None,
    show_progress: bool = True,
) -> dict[str, object]:
    """대여이력 CSV 목록을 읽어 EnvConfig가 사용할 profile JSON을 생성한다."""
    if master_csv is not None and not master_csv.is_file():
        raise ValueError(f"master_csv must be a file: {master_csv}")

    if station_ids:
        candidates = _candidates_from_station_ids(
            rental_paths=rental_paths,
            station_ids=station_ids,
            encoding=encoding,
            show_progress=show_progress,
        )
    else:
        candidates = find_station_candidates(
            rental_paths=rental_paths,
            station_keyword=station_keyword,
            station_count=station_count,
            encoding=encoding,
            show_progress=show_progress,
        )

    if not candidates:
        raise ValueError(f"No stations found for keyword or ids: {station_keyword}")

    station_index = {
        candidate.station_id: index for index, candidate in enumerate(candidates)
    }
    demand_totals = [[0 for _ in candidates] for _ in range(24)]
    return_totals = [[0 for _ in candidates] for _ in range(24)]
    active_dates: set[str] = set()

    for row in _iter_rental_rows(
        rental_paths,
        encoding=encoding,
        desc="집계",
        show_progress=show_progress,
    ):
        rent_station_id = row.get(RENT_STATION_ID_COLUMN, "").strip()
        if rent_station_id in station_index:
            rent_time = _parse_datetime(row.get(RENT_TIME_COLUMN, ""))
            if rent_time is not None:
                demand_totals[rent_time.hour][station_index[rent_station_id]] += 1
                active_dates.add(rent_time.date().isoformat())

        return_station_id = row.get(RETURN_STATION_ID_COLUMN, "").strip()
        if return_station_id in station_index:
            return_time = _parse_datetime(row.get(RETURN_TIME_COLUMN, ""))
            if return_time is not None:
                return_totals[return_time.hour][station_index[return_station_id]] += 1
                active_dates.add(return_time.date().isoformat())

    day_count = max(1, len(active_dates))
    demand_avg = _daily_average(demand_totals, day_count)
    return_avg = _daily_average(return_totals, day_count)
    profile_scale = scale if scale is not None else _auto_scale(
        demand_avg,
        return_avg,
        max_sample_high=max_sample_high,
    )
    master_lookup = _load_master_lookup(master_csv, encoding) if master_csv else {}

    payload: dict[str, object] = {
        "stations": [
            {
                "id": candidate.station_id,
                "name": candidate.name,
                **master_lookup.get(candidate.station_id, {}),
            }
            for candidate in candidates
        ],
        "demand_ranges_by_hour": _to_hourly_ranges(
            demand_avg,
            scale=profile_scale,
            max_sample_high=max_sample_high,
        ),
        "return_ranges_by_hour": _to_hourly_ranges(
            return_avg,
            scale=profile_scale,
            max_sample_high=max_sample_high,
        ),
        "metadata": {
            "source_files": [str(path) for path in rental_paths],
            "station_keyword": station_keyword,
            "day_count": day_count,
            "scale": profile_scale,
            "max_sample_high": max_sample_high,
            "candidate_counts": [
                {
                    "id": candidate.station_id,
                    "name": candidate.name,
                    "matched_rows": candidate.count,
                }
                for candidate in candidates
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def find_station_candidates(
    rental_paths: list[Path],
    station_keyword: str,
    station_count: int,
    encoding: str = "cp949",
    show_progress: bool = True,
) -> list[StationCandidate]:
    """대여소명에 keyword가 들어간 후보를 이용 건수 기준으로 고른다."""
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    keyword = station_keyword.strip()
    if not keyword:
        raise ValueError("station_keyword must not be empty")

    for row in _iter_rental_rows(
        rental_paths,
        encoding=encoding,
        desc="후보 검색",
        show_progress=show_progress,
    ):
        _add_candidate_match(
            row=row,
            keyword=keyword,
            id_column=RENT_STATION_ID_COLUMN,
            name_column=RENT_STATION_NAME_COLUMN,
            counts=counts,
            names=names,
        )
        _add_candidate_match(
            row=row,
            keyword=keyword,
            id_column=RETURN_STATION_ID_COLUMN,
            name_column=RETURN_STATION_NAME_COLUMN,
            counts=counts,
            names=names,
        )

    ranked = sorted(counts, key=lambda station_id: counts[station_id], reverse=True)
    return [
        StationCandidate(
            station_id=station_id,
            name=names.get(station_id, station_id),
            count=counts[station_id],
        )
        for station_id in ranked[:station_count]
    ]


def rental_csv_paths(rental_dir: Path) -> list[Path]:
    """대여이력 CSV 폴더에서 월별 CSV 파일 경로를 정렬해 반환한다."""
    if rental_dir.is_file():
        return [rental_dir]
    paths = sorted(path for path in rental_dir.glob("*.csv") if path.is_file())
    if not paths:
        raise ValueError(f"No rental CSV files found: {rental_dir}")
    return paths


def main(argv: list[str] | None = None) -> None:
    """명령행 인자를 받아 실제 데이터 profile JSON을 생성한다."""
    parser = ArgumentParser(description="Build Ddareungi real-data profile JSON.")
    parser.add_argument("--rental-dir", type=Path, required=True)
    parser.add_argument("--master-csv", type=Path)
    parser.add_argument("--station-keyword", default="마곡")
    parser.add_argument("--station-count", type=int, default=3)
    parser.add_argument("--station-ids", default="")
    parser.add_argument("--encoding", default="cp949")
    parser.add_argument("--max-sample-high", type=int, default=5)
    parser.add_argument("--scale", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data/magok_3station_profile.json"),
    )
    args = parser.parse_args(argv)

    station_ids = tuple(
        station_id.strip()
        for station_id in args.station_ids.split(",")
        if station_id.strip()
    )
    profile = build_profile_from_csvs(
        rental_paths=rental_csv_paths(args.rental_dir),
        output_path=args.output,
        station_keyword=args.station_keyword,
        station_count=args.station_count,
        station_ids=station_ids,
        master_csv=args.master_csv,
        encoding=args.encoding,
        max_sample_high=args.max_sample_high,
        scale=args.scale,
        show_progress=True,
    )
    stations = ", ".join(station["name"] for station in profile["stations"])
    print(f"profile_saved={args.output}")
    print(f"stations={stations}")
    print(f"day_count={profile['metadata']['day_count']}")
    print(f"scale={profile['metadata']['scale']}")


def _iter_rental_rows(
    rental_paths: Iterable[Path],
    encoding: str,
    desc: str,
    show_progress: bool,
) -> Iterable[dict[str, str]]:
    """여러 CSV 파일을 순서대로 열어 DictReader row를 하나씩 반환한다."""
    for path in rental_paths:
        with path.open("r", encoding=encoding, newline="") as file:
            reader = csv.DictReader(file)
            iterator = reader
            if show_progress and tqdm is not None:
                iterator = tqdm(reader, desc=f"{desc}: {path.name}", unit="rows")
            for row in iterator:
                yield row


def _add_candidate_match(
    row: dict[str, str],
    keyword: str,
    id_column: str,
    name_column: str,
    counts: dict[str, int],
    names: dict[str, str],
) -> None:
    """row의 대여소명이 keyword와 맞으면 후보 집계에 더한다."""
    station_id = row.get(id_column, "").strip()
    station_name = row.get(name_column, "").strip()
    if station_id and keyword in station_name:
        counts[station_id] = counts.get(station_id, 0) + 1
        names[station_id] = station_name


def _candidates_from_station_ids(
    rental_paths: list[Path],
    station_ids: tuple[str, ...],
    encoding: str,
    show_progress: bool,
) -> list[StationCandidate]:
    """직접 지정한 station id들의 이름과 출현 횟수를 CSV에서 찾는다."""
    wanted = set(station_ids)
    counts = {station_id: 0 for station_id in station_ids}
    names: dict[str, str] = {}
    for row in _iter_rental_rows(
        rental_paths,
        encoding=encoding,
        desc="ID 확인",
        show_progress=show_progress,
    ):
        for id_column, name_column in (
            (RENT_STATION_ID_COLUMN, RENT_STATION_NAME_COLUMN),
            (RETURN_STATION_ID_COLUMN, RETURN_STATION_NAME_COLUMN),
        ):
            station_id = row.get(id_column, "").strip()
            if station_id in wanted:
                counts[station_id] += 1
                names[station_id] = row.get(name_column, station_id).strip() or station_id
    return [
        StationCandidate(
            station_id=station_id,
            name=names.get(station_id, station_id),
            count=counts[station_id],
        )
        for station_id in station_ids
    ]


def _parse_datetime(raw_value: str) -> datetime | None:
    """CSV의 날짜 문자열을 datetime으로 변환하고 실패하면 None을 반환한다."""
    value = raw_value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _daily_average(hourly_totals: list[list[int]], day_count: int) -> list[list[float]]:
    """전체 집계량을 하루 평균 시간대별 건수로 변환한다."""
    return [[count / day_count for count in station_counts] for station_counts in hourly_totals]


def _auto_scale(
    demand_avg: list[list[float]],
    return_avg: list[list[float]],
    max_sample_high: int,
) -> float:
    """실제 평균 건수를 환경 capacity에 맞는 작은 샘플 범위로 축소할 배율을 정한다."""
    max_average = max(
        [value for hour_values in demand_avg + return_avg for value in hour_values],
        default=0.0,
    )
    if max_average <= 0:
        return 1.0
    return max_sample_high / max_average


def _to_hourly_ranges(
    averages: list[list[float]],
    scale: float,
    max_sample_high: int,
) -> dict[str, list[list[int]]]:
    """시간대별 평균 건수를 환경이 샘플링할 정수 범위로 변환한다."""
    ranges: dict[str, list[list[int]]] = {}
    for hour, station_averages in enumerate(averages):
        ranges[str(hour)] = [
            _scaled_range(value, scale=scale, max_sample_high=max_sample_high)
            for value in station_averages
        ]
    return ranges


def _scaled_range(value: float, scale: float, max_sample_high: int) -> list[int]:
    """평균값 하나를 [low, high] 샘플 범위로 변환한다."""
    scaled = value * scale
    low = max(0, min(max_sample_high, floor(scaled)))
    high = max(0, min(max_sample_high, ceil(scaled)))
    if value > 0 and high == 0:
        high = 1
    if low > high:
        low = high
    return [low, high]


def _load_master_lookup(master_csv: Path, encoding: str) -> dict[str, dict[str, object]]:
    """대여소 마스터 CSV에서 주소와 좌표를 station id 기준으로 읽는다."""
    lookup: dict[str, dict[str, object]] = {}
    with master_csv.open("r", encoding=encoding, newline="") as file:
        for row in csv.DictReader(file):
            station_id = row.get("대여소_ID", "").strip()
            if station_id:
                lookup[station_id] = {
                    "address1": row.get("주소1", "").strip(),
                    "address2": row.get("주소2", "").strip(),
                    "latitude": _to_float(row.get("위도", "")),
                    "longitude": _to_float(row.get("경도", "")),
                }
    return lookup


def _to_float(raw_value: str) -> float | None:
    """빈 문자열을 허용하면서 좌표 문자열을 float로 변환한다."""
    try:
        return float(raw_value)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
