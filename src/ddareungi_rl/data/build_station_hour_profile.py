"""실제 따릉이 CSV에서 선택 station의 시간대별 profile을 생성하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddareungi_rl.data.real_data import (
    DEFAULT_ENCODING,
    DEFAULT_MAGOK_STATION_IDS,
    build_station_hour_profile,
    write_candidates_csv,
    write_hourly_profile_csv,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    """profile 생성 CLI의 argument parser를 만든다."""
    parser = argparse.ArgumentParser(
        description="따릉이 대여이력 CSV를 streaming으로 읽어 선택 station profile을 만듭니다."
    )
    parser.add_argument("rental_csv", type=Path, help="대여이력 CSV 경로")
    parser.add_argument("master_csv", type=Path, help="대여소 마스터 CSV 경로")
    parser.add_argument(
        "--station-id",
        action="append",
        dest="station_ids",
        help="profile에 포함할 station ID. 여러 번 지정 가능",
    )
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help="CSV 인코딩")
    parser.add_argument("--max-rows", type=int, default=None, help="스캔할 최대 row 수")
    parser.add_argument("--start-date", default=None, help="포함할 시작 날짜 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="포함할 종료 날짜 YYYY-MM-DD")
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="월간 count를 toy 정수 범위로 줄일 때 나누는 스케일",
    )
    parser.add_argument(
        "--max-high",
        type=int,
        default=5,
        help="toy demand/return 샘플링 범위의 최대 상한",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("outputs/data/magok_3station_profile.json"),
        help="환경 주입용 profile JSON 저장 경로",
    )
    parser.add_argument(
        "--hourly-output",
        type=Path,
        default=Path("outputs/data/magok_3station_hourly_profile.csv"),
        help="사람 확인용 station-hour CSV 저장 경로",
    )
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=Path("outputs/data/magok_station_candidates.csv"),
        help="선택 station 총량 요약 CSV 저장 경로",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI 인자를 받아 station-hour profile 산출물을 생성한다."""
    args = build_parser().parse_args(argv)
    station_ids = tuple(args.station_ids or DEFAULT_MAGOK_STATION_IDS)
    profile = build_station_hour_profile(
        args.rental_csv,
        args.master_csv,
        station_ids=station_ids,
        encoding=args.encoding,
        max_rows=args.max_rows,
        scale=args.scale,
        max_high=args.max_high,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    write_json(profile, args.profile_output)
    write_hourly_profile_csv(profile, args.hourly_output)
    write_candidates_csv(profile, args.candidates_output)
    print(f"rows_scanned={profile['metadata']['rows_scanned']}")
    print(f"observed_days={profile['metadata']['observed_days']}")
    print(f"profile_output={args.profile_output}")
    print(f"hourly_output={args.hourly_output}")
    print(f"candidates_output={args.candidates_output}")


if __name__ == "__main__":
    main()
