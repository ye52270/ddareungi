import tempfile
import unittest
from pathlib import Path

from ddareungi_rl.baselines import LowStockPolicy, NoOpPolicy, RandomPolicy
from ddareungi_rl.config_loader import load_default_config
from ddareungi_rl.data_profile import load_profile
from ddareungi_rl.dqn import DQNConfig, evaluate_policy, train_dqn
from ddareungi_rl.env import DdareungiEnv, EnvConfig
from ddareungi_rl.profile_builder import build_daily_profile_from_csvs, build_profile_from_csvs


class SimpleProjectTest(unittest.TestCase):
    """단순화된 프로젝트의 핵심 흐름을 검증한다."""

    def test_default_config_loads_from_files(self):
        """기본 환경 설정과 샘플 데이터가 파일에서 로드되는지 확인한다."""
        config = load_default_config()

        self.assertEqual(config.station_names[0], "마곡나루역")
        self.assertEqual(config.initial_stock_min, 2)
        self.assertEqual(config.initial_stock_max, 8)
        self.assertEqual(len(config.demand_ranges), 24)
        self.assertEqual(len(config.return_ranges), 24)

    def test_config_rejects_invalid_hourly_ranges(self):
        """EnvConfig가 잘못된 시간대별 샘플 범위를 거부하는지 확인한다."""
        with self.assertRaises(ValueError):
            EnvConfig(
                station_names=("A", "B", "C"),
                demand_ranges={0: ((0, 1), (0, 1), (0, 1))},
                return_ranges={hour: ((0, 1), (0, 1), (0, 1)) for hour in range(24)},
            )

    def test_config_rejects_invalid_initial_stock_range(self):
        """초기 재고 범위가 대여소 용량을 넘으면 오류를 내는지 확인한다."""
        with self.assertRaises(ValueError):
            EnvConfig(
                station_names=("A", "B", "C"),
                demand_ranges={hour: ((0, 1), (0, 1), (0, 1)) for hour in range(24)},
                return_ranges={hour: ((0, 1), (0, 1), (0, 1)) for hour in range(24)},
                station_capacity=10,
                initial_stock_min=2,
                initial_stock_max=11,
            )

    def test_env_runs_one_episode(self):
        """환경이 reset 후 24 step episode를 끝까지 실행하는지 확인한다."""
        env = DdareungiEnv(seed=123)
        observation, info = env.reset(seed=123)
        steps = 0
        done = False

        while not done:
            observation, reward, terminated, truncated, info = env.step(0)
            steps += 1
            done = terminated or truncated

        self.assertEqual(steps, 24)
        self.assertEqual(len(observation), env.observation_space.shape[0])
        self.assertIn("unmet_demand", info)

    def test_baseline_evaluation_runs(self):
        """baseline policy를 여러 episode에서 평가할 수 있는지 확인한다."""
        env = DdareungiEnv(seed=123)
        result = evaluate_policy(env, RandomPolicy(seed=123), episodes=2, seed=123)

        self.assertIn("avg_reward", result)
        self.assertIn("avg_unmet_demand", result)
        self.assertIn("avg_rejected_returns", result)
        self.assertIn("avg_service_rate", result)

    def test_no_op_policy_keeps_truck_location(self):
        """NO-OP baseline이 현재 트럭 위치를 그대로 반환하는지 확인한다."""
        env = DdareungiEnv(seed=123)
        env.reset(seed=123)
        env.truck_location = 2

        self.assertEqual(NoOpPolicy().act(env), 2)

    def test_reward_penalizes_rejected_returns(self):
        """reward가 반납 실패도 벌점으로 반영하는지 확인한다."""
        config = EnvConfig(
            station_names=("A", "B", "C"),
            demand_ranges={hour: ((0, 0), (0, 0), (0, 0)) for hour in range(24)},
            return_ranges={hour: ((2, 2), (0, 0), (0, 0)) for hour in range(24)},
        )
        env = DdareungiEnv(config=config, seed=123)
        env.reset(seed=123)
        env.station_bikes = [10, 5, 5]
        env.truck_bikes = env.config.truck_capacity

        _, reward, _, _, info = env.step(0)

        self.assertEqual(info["rejected_returns"], 2)
        self.assertEqual(reward, -6.0)

    def test_low_stock_policy_selects_lowest_station(self):
        """Low-stock baseline이 재고가 가장 낮은 대여소를 고르는지 확인한다."""
        env = DdareungiEnv(seed=123)
        env.reset(seed=123)
        env.station_bikes = [5, 1, 3]

        self.assertEqual(LowStockPolicy().act(env), 1)

    def test_dqn_training_smoke(self):
        """DQN 학습이 짧은 설정에서 실행되는지 확인한다."""
        env = DdareungiEnv(seed=123)
        config = DQNConfig(
            episodes=2,
            batch_size=4,
            min_replay=4,
            target_update=4,
            epsilon_decay=10,
            hidden_size=8,
        )

        policy, metrics = train_dqn(env, config=config, seed=123)
        result = evaluate_policy(env, policy, episodes=1, seed=999)

        self.assertEqual(len(metrics), 2)
        self.assertIn("avg_reward", result)

    def test_profile_builder_creates_real_data_profile(self):
        """작은 CSV 샘플에서 real-profile JSON을 만들고 환경 설정으로 읽는다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rental_csv = root / "rentals.csv"
            output_path = root / "profile.json"
            rental_csv.write_text(
                "\n".join(
                    [
                        "자전거번호,대여일시,대여 대여소번호,대여 대여소명,대여거치대,"
                        "반납일시,반납대여소번호,반납대여소명,반납거치대,이용시간(분),"
                        "이용거리(M),생년,성별,이용자종류,대여대여소ID,반납대여소ID,자전거구분",
                        "SPB-1,2025-01-01 08:00:00,1,마곡나루역,0,"
                        "2025-01-01 09:00:00,2,마곡수명산,0,60,1000,1990,M,내국인,ST-1,ST-2,일반자전거",
                        "SPB-2,2025-01-02 08:30:00,1,마곡나루역,0,"
                        "2025-01-02 09:20:00,2,마곡수명산,0,50,900,1991,F,내국인,ST-1,ST-2,일반자전거",
                        "SPB-3,2025-01-02 18:00:00,2,마곡수명산,0,"
                        "2025-01-02 19:00:00,1,마곡나루역,0,60,800,1992,F,내국인,ST-2,ST-1,일반자전거",
                    ]
                ),
                encoding="utf-8",
            )

            profile = build_profile_from_csvs(
                rental_paths=[rental_csv],
                output_path=output_path,
                station_keyword="마곡",
                station_count=2,
                encoding="utf-8",
                show_progress=False,
            )
            config = load_profile(output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(len(profile["stations"]), 2)
            self.assertEqual(config.station_count, 2)
            self.assertEqual(config.demand_ranges[8][0][1], 5)

    def test_daily_profile_builder_keeps_date_hour_counts(self):
        """작은 CSV 샘플에서 날짜/시간대별 대여와 반납 count를 보존한다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rental_csv = root / "rentals.csv"
            output_path = root / "daily_profile.json"
            rental_csv.write_text(
                "\n".join(
                    [
                        "자전거번호,대여일시,대여 대여소번호,대여 대여소명,대여거치대,"
                        "반납일시,반납대여소번호,반납대여소명,반납거치대,이용시간(분),"
                        "이용거리(M),생년,성별,이용자종류,대여대여소ID,반납대여소ID,자전거구분",
                        "SPB-1,2025-01-01 08:00:00,1,마곡나루역,0,"
                        "2025-01-01 09:00:00,2,마곡수명산,0,60,1000,1990,M,내국인,ST-1,ST-2,일반자전거",
                        "SPB-2,2025-01-01 08:30:00,1,마곡나루역,0,"
                        "2025-01-01 09:20:00,2,마곡수명산,0,50,900,1991,F,내국인,ST-1,ST-2,일반자전거",
                        "SPB-3,2025-01-02 18:00:00,2,마곡수명산,0,"
                        "2025-01-02 19:00:00,1,마곡나루역,0,60,800,1992,F,내국인,ST-2,ST-1,일반자전거",
                    ]
                ),
                encoding="utf-8",
            )

            profile = build_daily_profile_from_csvs(
                rental_paths=[rental_csv],
                output_path=output_path,
                station_keyword="마곡",
                station_count=2,
                encoding="utf-8",
                show_progress=False,
            )

            self.assertEqual(profile["profile_kind"], "daily")
            self.assertEqual(profile["metadata"]["day_count"], 2)
            self.assertEqual(profile["metadata"]["observation_count"], 96)
            self.assertEqual(profile["daily_demand_counts"]["2025-01-01"][8][0], 2)
            self.assertEqual(profile["daily_return_counts"]["2025-01-01"][9][1], 2)
            self.assertEqual(profile["daily_demand_counts"]["2025-01-02"][18][1], 1)
            self.assertEqual(profile["daily_return_counts"]["2025-01-02"][19][0], 1)


if __name__ == "__main__":
    unittest.main()
