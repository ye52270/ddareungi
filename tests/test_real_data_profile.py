import tempfile
from pathlib import Path
import unittest

from ddareungi_rl.data.profile_loader import load_profile_config
from ddareungi_rl.data.real_data import (
    build_station_hour_profile,
    inspect_rental_csv,
    read_master_stations,
    write_json,
)
from ddareungi_rl.envs import ToyDdareungiEnv


class RealDataProfileTest(unittest.TestCase):
    """실제 따릉이 CSV profile helper의 작은 fixture 기반 테스트."""

    def test_inspect_rental_csv_reads_columns_and_samples(self):
        """inspection helper가 컬럼과 샘플 row를 반환하는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rental_csv, _ = self._write_fixture_files(Path(tmp_dir))

            inspection = inspect_rental_csv(rental_csv, sample_rows=1)

        self.assertIn("대여일시", inspection.columns)
        self.assertEqual(len(inspection.sampled_rows), 1)
        self.assertEqual(inspection.rows_scanned, 4)

    def test_read_master_stations_reads_coordinates(self):
        """마스터 CSV helper가 station 좌표를 읽는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, master_csv = self._write_fixture_files(Path(tmp_dir))

            stations = read_master_stations(master_csv)

        self.assertAlmostEqual(stations["ST-2031"].lat or 0.0, 37.566925)
        self.assertEqual(stations["ST-2033"].address2, "LG유플러스 마곡사옥")

    def test_build_station_hour_profile_creates_profile_payload(self):
        """station-hour 집계가 demand/return range와 station totals를 만드는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rental_csv, master_csv = self._write_fixture_files(Path(tmp_dir))

            profile = build_station_hour_profile(
                rental_csv,
                master_csv,
                station_ids=("ST-2031", "ST-2033", "ST-2049"),
                scale=1.0,
                max_high=5,
            )

        self.assertEqual(profile["metadata"]["rows_scanned"], 4)
        self.assertEqual(profile["metadata"]["observed_days"], 2)
        self.assertEqual(profile["monthly_demand_by_hour"]["8"], [2, 0, 0])
        self.assertEqual(profile["monthly_return_by_hour"]["18"], [0, 2, 0])
        self.assertEqual(profile["station_totals"]["ST-2031"]["rent"], 2)
        self.assertEqual(profile["station_totals"]["ST-2033"]["return"], 2)
        self.assertEqual(profile["demand_ranges_by_hour"]["8"][0], [0, 2])

    def test_build_station_hour_profile_can_filter_date_range(self):
        """날짜 범위를 지정하면 해당 기간의 event만 profile에 반영되는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rental_csv, master_csv = self._write_fixture_files(Path(tmp_dir))

            profile = build_station_hour_profile(
                rental_csv,
                master_csv,
                station_ids=("ST-2031", "ST-2033", "ST-2049"),
                start_date="2025-12-02",
                end_date="2025-12-02",
            )

        self.assertEqual(profile["metadata"]["observed_days"], 1)
        self.assertEqual(profile["monthly_demand_by_hour"]["8"], [0, 0, 0])
        self.assertEqual(profile["monthly_demand_by_hour"]["9"], [0, 0, 1])
        self.assertEqual(profile["monthly_return_by_hour"]["20"], [1, 0, 0])

    def test_load_profile_config_runs_environment_episode(self):
        """profile JSON을 ToyDdareungiConfig로 변환해 episode를 실행할 수 있는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rental_csv, master_csv = self._write_fixture_files(tmp_path)
            profile = build_station_hour_profile(
                rental_csv,
                master_csv,
                station_ids=("ST-2031", "ST-2033", "ST-2049"),
            )
            profile_path = tmp_path / "profile.json"
            write_json(profile, profile_path)

            config = load_profile_config(profile_path)
            env = ToyDdareungiEnv(config=config, seed=123)
            observation, _ = env.reset(seed=123)
            for _ in range(config.episode_steps):
                observation, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    break

        self.assertEqual(config.station_count, 3)
        self.assertEqual(len(observation), env.observation_size)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue(truncated)

    def _write_fixture_files(self, tmp_path: Path) -> tuple[Path, Path]:
        """테스트용 작은 대여이력/마스터 CSV 파일을 만든다."""
        rental_csv = tmp_path / "rental.csv"
        master_csv = tmp_path / "master.csv"
        rental_rows = [
            {
                "자전거번호": "SPB-1",
                "대여일시": "2025-12-01 08:00:00",
                "대여 대여소명": "마곡나루역 2번 출구",
                "대여대여소ID": "ST-2031",
                "반납일시": "2025-12-01 18:00:00",
                "반납대여소명": "LG유플러스 마곡사옥",
                "반납대여소ID": "ST-2033",
            },
            {
                "자전거번호": "SPB-2",
                "대여일시": "2025-12-01 08:10:00",
                "대여 대여소명": "마곡나루역 2번 출구",
                "대여대여소ID": "ST-2031",
                "반납일시": "2025-12-01 18:20:00",
                "반납대여소명": "LG유플러스 마곡사옥",
                "반납대여소ID": "ST-2033",
            },
            {
                "자전거번호": "SPB-3",
                "대여일시": "2025-12-02 09:00:00",
                "대여 대여소명": "마곡수명산 1-2단지",
                "대여대여소ID": "ST-2049",
                "반납일시": "2025-12-02 20:00:00",
                "반납대여소명": "마곡나루역 2번 출구",
                "반납대여소ID": "ST-2031",
            },
            {
                "자전거번호": "SPB-4",
                "대여일시": "2025-12-02 09:30:00",
                "대여 대여소명": "다른 대여소",
                "대여대여소ID": "ST-0000",
                "반납일시": "2025-12-02 21:00:00",
                "반납대여소명": "다른 대여소",
                "반납대여소ID": "ST-0000",
            },
        ]
        master_rows = [
            ["대여소_ID", "주소1", "주소2", "위도", "경도"],
            ["ST-2031", "서울특별시 강서구", "마곡나루역 2번 출구", "37.566925", "126.827438"],
            ["ST-2033", "서울특별시 강서구", "LG유플러스 마곡사옥", "37.561337", "126.8339"],
            ["ST-2049", "서울특별시 강서구", "마곡수명산 1-2단지", "37.555309", "126.829857"],
        ]

        rental_csv.write_text(self._csv_text(rental_rows), encoding="cp949")
        master_csv.write_text(
            "\n".join(",".join(row) for row in master_rows) + "\n",
            encoding="cp949",
        )
        return rental_csv, master_csv

    def _csv_text(self, rows: list[dict[str, str]]) -> str:
        """dict row 목록을 테스트용 CSV 문자열로 변환한다."""
        columns = list(rows[0].keys())
        lines = [",".join(columns)]
        for row in rows:
            lines.append(",".join(row[column] for column in columns))
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
