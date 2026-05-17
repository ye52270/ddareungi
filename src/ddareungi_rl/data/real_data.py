"""실제 따릉이 CSV를 작은 toy profile로 집계하는 공통 로직."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from math import ceil, floor
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ENCODING = "cp949"
DEFAULT_MAGOK_STATION_IDS = ("ST-2031", "ST-2033", "ST-2049")
RENT_TIME_COLUMN = "대여일시"
RENT_STATION_ID_COLUMN = "대여대여소ID"
RENT_STATION_NAME_COLUMN = "대여 대여소명"
RETURN_TIME_COLUMN = "반납일시"
RETURN_STATION_ID_COLUMN = "반납대여소ID"
RETURN_STATION_NAME_COLUMN = "반납대여소명"
MASTER_STATION_ID_COLUMN = "대여소_ID"
MASTER_ADDRESS_1_COLUMN = "주소1"
MASTER_ADDRESS_2_COLUMN = "주소2"
MASTER_LAT_COLUMN = "위도"
MASTER_LON_COLUMN = "경도"


@dataclass(frozen=True)
class StationMetadata:
    """마스터 CSV에서 읽은 대여소의 최소 메타데이터를 담는다."""

    station_id: str
    address1: str
    address2: str
    lat: float | None
    lon: float | None


@dataclass(frozen=True)
class RentalCsvInspection:
    """대여이력 CSV의 schema와 샘플 확인 결과를 담는다."""

    path: str
    encoding: str
    columns: list[str]
    sampled_rows: list[dict[str, str]]
    rows_scanned: int
    stopped_early: bool


def normalize_text(value: object) -> str:
    """CSV cell 값을 비교 가능한 문자열로 정리한다."""
    if value is None:
        return ""
    return str(value).strip()


def parse_float(value: object) -> float | None:
    """숫자 문자열을 float로 변환하고 실패하면 None을 반환한다."""
    text = normalize_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_hour(value: object) -> int | None:
    """대여/반납 일시 문자열에서 0~23 hour를 추출한다."""
    text = normalize_text(value)
    if not text or text == r"\N":
        return None
    try:
        return datetime.fromisoformat(text).hour
    except ValueError:
        return None


def parse_date_key(value: object) -> str | None:
    """대여/반납 일시 문자열에서 YYYY-MM-DD 날짜 key를 추출한다."""
    text = normalize_text(value)
    if not text or text == r"\N":
        return None
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return None


def date_in_range(
    date_key: str | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> bool:
    """YYYY-MM-DD date key가 선택한 날짜 범위에 포함되는지 반환한다."""
    if date_key is None:
        return False
    if start_date is not None and date_key < start_date:
        return False
    if end_date is not None and date_key > end_date:
        return False
    return True


def read_master_stations(
    master_csv_path: Path,
    encoding: str = DEFAULT_ENCODING,
) -> dict[str, StationMetadata]:
    """대여소 마스터 CSV를 station ID 기준 metadata dict로 읽는다."""
    stations: dict[str, StationMetadata] = {}
    with master_csv_path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            station_id = normalize_text(row.get(MASTER_STATION_ID_COLUMN))
            if not station_id:
                continue
            stations[station_id] = StationMetadata(
                station_id=station_id,
                address1=normalize_text(row.get(MASTER_ADDRESS_1_COLUMN)),
                address2=normalize_text(row.get(MASTER_ADDRESS_2_COLUMN)),
                lat=parse_float(row.get(MASTER_LAT_COLUMN)),
                lon=parse_float(row.get(MASTER_LON_COLUMN)),
            )
    return stations


def inspect_rental_csv(
    rental_csv_path: Path,
    encoding: str = DEFAULT_ENCODING,
    sample_rows: int = 5,
    max_rows: int | None = None,
) -> RentalCsvInspection:
    """대여이력 CSV의 컬럼, 샘플 row, 스캔 row 수를 확인한다."""
    sampled_rows: list[dict[str, str]] = []
    rows_scanned = 0

    with rental_csv_path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows_scanned += 1
            if len(sampled_rows) < sample_rows:
                sampled_rows.append({key: normalize_text(value) for key, value in row.items()})
            if max_rows is not None and rows_scanned >= max_rows:
                break

    return RentalCsvInspection(
        path=str(rental_csv_path),
        encoding=encoding,
        columns=columns,
        sampled_rows=sampled_rows,
        rows_scanned=rows_scanned,
        stopped_early=max_rows is not None and rows_scanned >= max_rows,
    )


def empty_hour_station_matrix(station_count: int) -> dict[str, list[int]]:
    """24시간 x station_count 형태의 0 count matrix를 만든다."""
    return {str(hour): [0 for _ in range(station_count)] for hour in range(24)}


def count_to_sampling_range(
    monthly_count: int,
    observed_days: int,
    scale: float = 2.0,
    max_high: int = 5,
) -> tuple[int, int]:
    """월간 시간대 count를 toy 환경용 작은 정수 샘플링 범위로 변환한다."""
    if monthly_count <= 0 or observed_days <= 0:
        return (0, 0)
    daily_mean = monthly_count / observed_days
    scaled_mean = daily_mean / scale
    low = max(0, floor(scaled_mean * 0.7))
    high = min(max_high, ceil(scaled_mean * 1.3))
    if high == 0:
        high = 1
    if low > high:
        low = high
    return (low, high)


def ranges_from_counts(
    counts_by_hour: dict[str, list[int]],
    observed_days: int,
    scale: float = 2.0,
    max_high: int = 5,
) -> dict[str, list[list[int]]]:
    """시간대별 월간 count matrix를 toy 샘플링 범위 matrix로 변환한다."""
    return {
        hour: [
            list(count_to_sampling_range(count, observed_days, scale, max_high))
            for count in counts
        ]
        for hour, counts in counts_by_hour.items()
    }


def daily_means_from_counts(
    counts_by_hour: dict[str, list[int]],
    observed_days: int,
) -> dict[str, list[float]]:
    """시간대별 월간 count matrix를 일평균 count matrix로 변환한다."""
    divisor = max(1, observed_days)
    return {
        hour: [round(count / divisor, 3) for count in counts]
        for hour, counts in counts_by_hour.items()
    }


def build_station_hour_profile(
    rental_csv_path: Path,
    master_csv_path: Path,
    station_ids: Iterable[str] = DEFAULT_MAGOK_STATION_IDS,
    encoding: str = DEFAULT_ENCODING,
    max_rows: int | None = None,
    scale: float = 2.0,
    max_high: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """선택 station의 시간대별 대여/반납 profile JSON payload를 만든다."""
    station_id_list = [normalize_text(station_id) for station_id in station_ids]
    station_index = {station_id: index for index, station_id in enumerate(station_id_list)}
    master_stations = read_master_stations(master_csv_path, encoding=encoding)
    demand_by_hour = empty_hour_station_matrix(len(station_id_list))
    return_by_hour = empty_hour_station_matrix(len(station_id_list))
    rent_totals = {station_id: 0 for station_id in station_id_list}
    return_totals = {station_id: 0 for station_id in station_id_list}
    station_names = {station_id: station_id for station_id in station_id_list}
    observed_dates: set[str] = set()
    rows_scanned = 0
    bad_time_rows = 0
    missing_station_id_rows = 0

    with rental_csv_path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows_scanned += 1
            rent_id = normalize_text(row.get(RENT_STATION_ID_COLUMN))
            return_id = normalize_text(row.get(RETURN_STATION_ID_COLUMN))
            if not rent_id and not return_id:
                missing_station_id_rows += 1

            rent_date = parse_date_key(row.get(RENT_TIME_COLUMN))
            if date_in_range(rent_date, start_date=start_date, end_date=end_date):
                observed_dates.add(rent_date)

            rent_hour = parse_hour(row.get(RENT_TIME_COLUMN))
            if (
                rent_id in station_index
                and rent_hour is not None
                and date_in_range(rent_date, start_date=start_date, end_date=end_date)
            ):
                index = station_index[rent_id]
                demand_by_hour[str(rent_hour)][index] += 1
                rent_totals[rent_id] += 1
                station_names[rent_id] = normalize_text(row.get(RENT_STATION_NAME_COLUMN)) or rent_id
            elif rent_id in station_index and rent_hour is None:
                bad_time_rows += 1

            return_date = parse_date_key(row.get(RETURN_TIME_COLUMN))
            if date_in_range(return_date, start_date=start_date, end_date=end_date):
                observed_dates.add(return_date)

            return_hour = parse_hour(row.get(RETURN_TIME_COLUMN))
            if (
                return_id in station_index
                and return_hour is not None
                and date_in_range(return_date, start_date=start_date, end_date=end_date)
            ):
                index = station_index[return_id]
                return_by_hour[str(return_hour)][index] += 1
                return_totals[return_id] += 1
                station_names[return_id] = (
                    normalize_text(row.get(RETURN_STATION_NAME_COLUMN)) or return_id
                )
            elif return_id in station_index and return_hour is None:
                bad_time_rows += 1

            if max_rows is not None and rows_scanned >= max_rows:
                break

    observed_days = max(1, len(observed_dates))
    stations = [
        station_payload(station_id, station_names[station_id], master_stations)
        for station_id in station_id_list
    ]

    return {
        "metadata": {
            "source_rental_csv": str(rental_csv_path),
            "source_master_csv": str(master_csv_path),
            "encoding": encoding,
            "rows_scanned": rows_scanned,
            "stopped_early": max_rows is not None and rows_scanned >= max_rows,
            "observed_days": observed_days,
            "start_date": start_date,
            "end_date": end_date,
            "bad_time_rows_for_selected_stations": bad_time_rows,
            "missing_station_id_rows": missing_station_id_rows,
            "normalization": (
                "monthly station-hour counts are divided by observed_days and "
                f"scale={scale}, then converted to integer sampling ranges capped at {max_high}"
            ),
        },
        "stations": stations,
        "monthly_demand_by_hour": demand_by_hour,
        "monthly_return_by_hour": return_by_hour,
        "daily_mean_demand_by_hour": daily_means_from_counts(demand_by_hour, observed_days),
        "daily_mean_return_by_hour": daily_means_from_counts(return_by_hour, observed_days),
        "demand_ranges_by_hour": ranges_from_counts(
            demand_by_hour,
            observed_days,
            scale=scale,
            max_high=max_high,
        ),
        "return_ranges_by_hour": ranges_from_counts(
            return_by_hour,
            observed_days,
            scale=scale,
            max_high=max_high,
        ),
        "station_totals": {
            station_id: {
                "rent": rent_totals[station_id],
                "return": return_totals[station_id],
                "imbalance_rent_minus_return": rent_totals[station_id]
                - return_totals[station_id],
            }
            for station_id in station_id_list
        },
    }


def station_payload(
    station_id: str,
    station_name: str,
    master_stations: dict[str, StationMetadata],
) -> dict[str, Any]:
    """station ID, 이름, 마스터 좌표를 JSON 직렬화 가능한 dict로 만든다."""
    metadata = master_stations.get(station_id)
    return {
        "id": station_id,
        "name": station_name,
        "address1": metadata.address1 if metadata else "",
        "address2": metadata.address2 if metadata else "",
        "lat": metadata.lat if metadata else None,
        "lon": metadata.lon if metadata else None,
    }


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """dict payload를 UTF-8 JSON 파일로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_hourly_profile_csv(profile: dict[str, Any], output_path: Path) -> None:
    """profile의 시간대별 월간 count를 사람이 확인하기 쉬운 CSV로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stations = profile["stations"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["hour", "station_id", "station_name", "rent_count", "return_count"])
        for hour in range(24):
            hour_key = str(hour)
            for index, station in enumerate(stations):
                writer.writerow(
                    [
                        hour,
                        station["id"],
                        station["name"],
                        profile["monthly_demand_by_hour"][hour_key][index],
                        profile["monthly_return_by_hour"][hour_key][index],
                    ]
                )


def write_candidates_csv(profile: dict[str, Any], output_path: Path) -> None:
    """선택 station의 총 대여/반납 imbalance 요약 CSV를 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["station_id", "station_name", "rent", "return", "rent_minus_return"])
        for station in profile["stations"]:
            totals = profile["station_totals"][station["id"]]
            writer.writerow(
                [
                    station["id"],
                    station["name"],
                    totals["rent"],
                    totals["return"],
                    totals["imbalance_rent_minus_return"],
                ]
            )
