"""대용량 따릉이 대여이력 CSV의 schema와 샘플을 확인하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddareungi_rl.data.real_data import (
    DEFAULT_ENCODING,
    inspect_rental_csv,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    """inspect CLI의 argument parser를 만든다."""
    parser = argparse.ArgumentParser(
        description="따릉이 대여이력 CSV의 컬럼, 샘플 row, 스캔 row 수를 확인합니다."
    )
    parser.add_argument("rental_csv", type=Path, help="대여이력 CSV 경로")
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, help="CSV 인코딩")
    parser.add_argument("--sample-rows", type=int, default=5, help="저장할 샘플 row 수")
    parser.add_argument("--max-rows", type=int, default=None, help="스캔할 최대 row 수")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data/rental_csv_inspection.json"),
        help="inspection JSON 저장 경로",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI 인자를 받아 대여이력 CSV inspection 결과를 저장하고 요약 출력한다."""
    args = build_parser().parse_args(argv)
    inspection = inspect_rental_csv(
        args.rental_csv,
        encoding=args.encoding,
        sample_rows=args.sample_rows,
        max_rows=args.max_rows,
    )
    payload = {
        "path": inspection.path,
        "encoding": inspection.encoding,
        "columns": inspection.columns,
        "sampled_rows": inspection.sampled_rows,
        "rows_scanned": inspection.rows_scanned,
        "stopped_early": inspection.stopped_early,
    }
    write_json(payload, args.output)
    print(f"rows_scanned={inspection.rows_scanned}")
    print(f"columns={len(inspection.columns)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
