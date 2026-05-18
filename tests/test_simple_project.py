import tempfile
import unittest
from pathlib import Path

from ddareungi_rl.baselines import LowStockPolicy, NoOpPolicy, RandomPolicy
from ddareungi_rl.config_loader import load_default_config
from ddareungi_rl.dashboard import save_experiment_dashboard
from ddareungi_rl.data_profile import load_profile
from ddareungi_rl.dqn import (
    DQNConfig,
    evaluate_policy,
    train_double_dqn,
    train_dqn,
    train_dueling_dqn,
)
from ddareungi_rl.env import DdareungiEnv, EnvConfig
from ddareungi_rl.experiment_log import append_dqn_experiment_log, read_experiment_log
from ddareungi_rl.profile_builder import build_daily_profile_from_csvs, build_profile_from_csvs
from ddareungi_rl.reporting import (
    save_algorithm_comparison_from_reports,
    save_baseline_vs_dqn_csv,
    save_multiseed_reports,
    save_policy_trace_reports,
    save_training_history_csv,
)


class SimpleProjectTest(unittest.TestCase):
    """단순화된 프로젝트의 핵심 흐름을 검증한다."""

    def test_default_config_loads_from_files(self):
        """기본 환경 설정과 샘플 데이터가 파일에서 로드되는지 확인한다."""
        config = load_default_config()

        self.assertEqual(config.station_names[0], "마곡나루역")
        self.assertEqual(config.initial_stock_min, 1)
        self.assertEqual(config.initial_stock_max, 5)
        self.assertEqual(len(config.demand_ranges), 24)
        self.assertEqual(len(config.return_ranges), 24)
        self.assertTrue(config.traffic_enabled)
        self.assertEqual(config.traffic_factors[8], 1.5)
        self.assertEqual(config.traffic_factors[18], 1.6)

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

    def test_config_rejects_invalid_traffic_factors(self):
        """traffic factor 개수가 episode 길이와 다르면 오류를 내는지 확인한다."""
        with self.assertRaises(ValueError):
            EnvConfig(
                station_names=("A", "B", "C"),
                demand_ranges={hour: ((0, 1), (0, 1), (0, 1)) for hour in range(24)},
                return_ranges={hour: ((0, 1), (0, 1), (0, 1)) for hour in range(24)},
                traffic_enabled=True,
                traffic_factors=(1.0, 1.5),
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

    def test_traffic_factor_adjusts_movement_cost(self):
        """출퇴근 시간 traffic factor가 트럭 이동비용에 반영되는지 확인한다."""
        config = EnvConfig(
            station_names=("A", "B"),
            demand_ranges={hour: ((0, 0), (0, 0)) for hour in range(24)},
            return_ranges={hour: ((0, 0), (0, 0)) for hour in range(24)},
            initial_stock_min=5,
            initial_stock_max=5,
            traffic_enabled=True,
            traffic_factors=tuple(1.5 if hour == 8 else 1.0 for hour in range(24)),
        )
        env = DdareungiEnv(config=config, seed=123)
        env.reset(seed=123)
        env.time_step = 8

        _, reward, _, _, info = env.step(1)

        self.assertEqual(info["traffic_factor"], 1.5)
        self.assertEqual(info["movement_cost"], 1.5)
        self.assertEqual(reward, -1.5)

    def test_observation_includes_expected_demand(self):
        """DQN observation에 대여소별 현재 시간대 예상 수요가 포함되는지 확인한다."""
        config = EnvConfig(
            station_names=("A", "B", "C"),
            demand_ranges={hour: ((0, 0), (0, 0), (0, 0)) for hour in range(24)},
            return_ranges={hour: ((0, 0), (0, 0), (0, 0)) for hour in range(24)},
            daily_dates=("2025-01-01",),
            daily_demand_counts=tuple(
                tuple((2, 4, 6) if hour == 0 else (0, 0, 0) for hour in range(24))
                for _ in range(1)
            ),
            daily_return_counts=tuple(
                tuple((0, 0, 0) for hour in range(24))
                for _ in range(1)
            ),
            station_capacity=10,
            initial_stock_min=5,
            initial_stock_max=5,
        )
        env = DdareungiEnv(config=config, seed=123)
        observation, _ = env.reset(seed=123, options={"daily_index": 0})

        self.assertEqual(len(observation), config.station_count * 2 + 3)
        self.assertEqual(list(observation[config.station_count: config.station_count * 2]), [0.2, 0.4, 0.6])

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

    def test_dueling_dqn_training_smoke(self):
        """Dueling DQN 학습이 DQN과 같은 환경/평가 흐름에서 실행되는지 확인한다."""
        env = DdareungiEnv(seed=123)
        config = DQNConfig(
            episodes=2,
            batch_size=4,
            min_replay=4,
            target_update=4,
            epsilon_decay=10,
            hidden_size=8,
        )

        policy, metrics = train_dueling_dqn(env, config=config, seed=123)
        result = evaluate_policy(env, policy, episodes=1, seed=999)

        self.assertEqual(len(metrics), 2)
        self.assertIn("avg_reward", result)

    def test_double_dqn_training_smoke(self):
        """Double DQN 학습이 DQN과 같은 환경/평가 흐름에서 실행되는지 확인한다."""
        env = DdareungiEnv(seed=123)
        config = DQNConfig(
            episodes=2,
            batch_size=4,
            min_replay=4,
            target_update=4,
            epsilon_decay=10,
            hidden_size=8,
        )

        policy, metrics = train_double_dqn(env, config=config, seed=123)
        result = evaluate_policy(env, policy, episodes=1, seed=999)

        self.assertEqual(len(metrics), 2)
        self.assertIn("avg_reward", result)

    def test_dqn_experiment_log_records_parameters(self):
        """DQN 실험 로그가 parameter와 평가 결과를 JSONL로 저장하는지 확인한다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_path = root / "dqn_runs.jsonl"
            config = DQNConfig(episodes=3, hidden_size=8)
            env = DdareungiEnv(seed=123)
            env.reset(seed=123)

            append_dqn_experiment_log(
                config=config,
                env=env,
                eval_episodes=2,
                eval_result={"avg_reward": -1.0, "avg_unmet_demand": 0.0},
                last_training_metric={"episode": 3.0, "reward": -1.0},
                model_path=root / "model.pt",
                curve_path=root / "curve.png",
                profile_path=root / "profile.json",
                log_path=log_path,
            )
            records = read_experiment_log(log_path)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["train_config"]["episodes"], 3)
            self.assertEqual(records[0]["train_config"]["hidden_size"], 8)
            self.assertEqual(records[0]["eval_episodes"], 2)
            self.assertEqual(records[0]["observation_size"], env.observation_space.shape[0])

    def test_report_csv_outputs_are_created(self):
        """논문식 report CSV 파일이 안정적인 schema로 저장되는지 확인한다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_path = root / "baseline_vs_dqn.csv"
            history_path = root / "dqn_training_history.csv"

            save_baseline_vs_dqn_csv(
                {
                    "low-stock": {
                        "avg_reward": -3.0,
                        "avg_unmet_demand": 1.0,
                        "avg_rejected_returns": 0.0,
                        "avg_movement_cost": 2.0,
                        "avg_service_rate": 0.9,
                        "same_location_rate": 0.1,
                    },
                    "dqn": {
                        "avg_reward": -2.0,
                        "avg_unmet_demand": 0.5,
                        "avg_rejected_returns": 0.0,
                        "avg_movement_cost": 1.0,
                        "avg_service_rate": 0.95,
                        "same_location_rate": 0.2,
                    },
                },
                output_path=comparison_path,
            )
            save_training_history_csv(
                [
                    {
                        "episode": 1.0,
                        "reward": -2.0,
                        "unmet_demand": 0.0,
                        "rejected_returns": 0.0,
                        "movement_cost": 1.0,
                        "epsilon": 0.9,
                        "loss": 0.1,
                    }
                ],
                output_path=history_path,
            )

            self.assertIn("policy,avg_reward", comparison_path.read_text(encoding="utf-8"))
            self.assertIn("low-stock,-3.0", comparison_path.read_text(encoding="utf-8"))
            self.assertIn("episode,reward", history_path.read_text(encoding="utf-8"))

    def test_dashboard_html_is_created_from_reports(self):
        """저장된 CSV/JSON report로 한 장짜리 HTML dashboard를 생성한다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_dir = root / "reports"
            figure_dir = root / "figures"
            report_dir.mkdir()
            figure_dir.mkdir()
            (report_dir / "experiment_config.json").write_text(
                """
                {
                  "environment": {
                    "station_names": ["A", "B"],
                    "station_count": 2,
                    "episode_steps": 24,
                    "profile_path": "profile.json",
                    "daily_profile_dates": 10
                  },
                  "mdp": {
                    "state": ["station_bikes", "expected_demand"],
                    "action": "next station",
                    "reward": "-10 * unmet_demand"
                  },
                  "dqn_config": {"episodes": 3}
                }
                """,
                encoding="utf-8",
            )
            save_baseline_vs_dqn_csv(
                {
                    "low-stock": {
                        "avg_reward": -3.0,
                        "avg_unmet_demand": 1.0,
                        "avg_rejected_returns": 0.5,
                        "avg_movement_cost": 2.0,
                        "avg_service_rate": 0.9,
                        "same_location_rate": 0.1,
                    },
                    "dqn": {
                        "avg_reward": -2.0,
                        "avg_unmet_demand": 0.5,
                        "avg_rejected_returns": 0.2,
                        "avg_movement_cost": 1.0,
                        "avg_service_rate": 0.95,
                        "same_location_rate": 0.2,
                    },
                },
                output_path=report_dir / "baseline_vs_dqn.csv",
            )
            save_training_history_csv(
                [
                    {
                        "episode": 1.0,
                        "reward": -2.0,
                        "unmet_demand": 0.0,
                        "rejected_returns": 0.0,
                        "movement_cost": 1.0,
                        "epsilon": 0.9,
                        "loss": 0.1,
                    }
                ],
                output_path=report_dir / "dqn_training_history.csv",
            )
            (report_dir / "action_distribution.csv").write_text(
                "policy,action,station_name,count,ratio\n"
                "dqn,0,A,10,0.5\n"
                "dqn,1,B,10,0.5\n",
                encoding="utf-8",
            )
            (report_dir / "dqn_evaluation_episodes.csv").write_text(
                "policy,episode,date,reward,served_demand,unmet_demand,"
                "rejected_returns,movement_cost,service_rate,same_location_steps\n"
                "dqn,1,2025-01-01,-2,10,0,0,1,1.0,2\n",
                encoding="utf-8",
            )
            (report_dir / "algorithm_comparison.csv").write_text(
                "algorithm,avg_reward,avg_unmet_demand,avg_rejected_returns,"
                "avg_movement_cost,avg_service_rate,same_location_rate\n"
                "low-stock,-3,1,0.5,2,0.9,0.1\n"
                "dqn,-2,0.5,0.2,1,0.95,0.2\n",
                encoding="utf-8",
            )

            dashboard_path = save_experiment_dashboard(
                output_path=report_dir / "experiment_dashboard.html",
                report_dir=report_dir,
                figure_dir=figure_dir,
            )
            html = dashboard_path.read_text(encoding="utf-8")

            self.assertIn("따릉이 RL Experiment Dashboard", html)
            self.assertIn("DQN Avg Reward", html)
            self.assertIn("전체 알고리즘 비교", html)
            self.assertIn("low-stock 대비 reward", html)

    def test_multiseed_reports_are_created(self):
        """seed별 DQN 평가 결과와 summary CSV를 저장한다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_paths = save_multiseed_reports(
                [
                    {
                        "seed": 42.0,
                        "avg_reward": -2.0,
                        "avg_unmet_demand": 0.5,
                        "avg_rejected_returns": 0.1,
                        "avg_movement_cost": 1.0,
                        "avg_service_rate": 0.95,
                        "same_location_rate": 0.2,
                    },
                    {
                        "seed": 142.0,
                        "avg_reward": -4.0,
                        "avg_unmet_demand": 1.5,
                        "avg_rejected_returns": 0.3,
                        "avg_movement_cost": 2.0,
                        "avg_service_rate": 0.9,
                        "same_location_rate": 0.4,
                    },
                ],
                runs_path=root / "runs.csv",
                summary_path=root / "summary.csv",
            )

            runs_text = report_paths["runs"].read_text(encoding="utf-8")
            summary_text = report_paths["summary"].read_text(encoding="utf-8")

            self.assertIn("seed,avg_reward", runs_text)
            self.assertIn("42.0,-2.0", runs_text)
            self.assertIn("avg_reward,-3.0", summary_text)

    def test_algorithm_comparison_collects_saved_results(self):
        """저장된 알고리즘별 결과 CSV를 하나의 비교표로 모은다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_baseline_vs_dqn_csv(
                {
                    "low-stock": {
                        "avg_reward": -3.0,
                        "avg_unmet_demand": 1.0,
                        "avg_rejected_returns": 0.5,
                        "avg_movement_cost": 2.0,
                        "avg_service_rate": 0.9,
                        "same_location_rate": 0.1,
                    },
                    "dqn": {
                        "avg_reward": -2.0,
                        "avg_unmet_demand": 0.5,
                        "avg_rejected_returns": 0.2,
                        "avg_movement_cost": 1.0,
                        "avg_service_rate": 0.95,
                        "same_location_rate": 0.2,
                    },
                },
                output_path=root / "baseline_vs_dqn.csv",
            )
            save_baseline_vs_dqn_csv(
                {
                    "double_dqn": {
                        "avg_reward": -1.5,
                        "avg_unmet_demand": 0.4,
                        "avg_rejected_returns": 0.2,
                        "avg_movement_cost": 1.0,
                        "avg_service_rate": 0.96,
                        "same_location_rate": 0.2,
                    },
                },
                output_path=root / "double_dqn_vs_baseline.csv",
            )

            comparison_path = save_algorithm_comparison_from_reports(
                report_dir=root,
                output_path=root / "algorithm_comparison.csv",
            )
            text = comparison_path.read_text(encoding="utf-8")

            self.assertIn("algorithm,avg_reward", text)
            self.assertIn("low-stock,-3.0", text)
            self.assertIn("dqn,-2.0", text)
            self.assertIn("double_dqn,-1.5", text)

    def test_policy_trace_reports_are_algorithm_specific(self):
        """알고리즘별 평가 trace가 서로 덮어쓰지 않는 파일명으로 저장되는지 확인한다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = {
                "action_counts": {0: 2, 1: 1},
                "episodes": [
                    {
                        "policy": "double_dqn",
                        "episode": 1,
                        "date": "2025-01-01",
                        "reward": -1.0,
                        "served_demand": 10,
                        "unmet_demand": 0,
                        "rejected_returns": 0,
                        "movement_cost": 1,
                        "service_rate": 1.0,
                        "same_location_steps": 1,
                    }
                ],
                "steps": [
                    {
                        "policy": "double_dqn",
                        "episode": 1,
                        "date": "2025-01-01",
                        "time_step": 1,
                        "action": 0,
                        "previous_truck_location": 0,
                        "truck_location": 0,
                        "truck_bikes": 5,
                        "reward": -1.0,
                        "served_demand": 10,
                        "unmet_demand": 0,
                        "rejected_returns": 0,
                        "movement_cost": 1,
                        "moved_bikes": 0,
                        "station_bikes": "5|5",
                        "demand": "1|0",
                        "returns": "0|0",
                    }
                ],
            }

            paths = save_policy_trace_reports(
                policy_name="double_dqn",
                station_names=["A", "B"],
                evaluation_report=report,
                report_dir=root,
            )

            self.assertEqual(paths["action_distribution"].name, "double_dqn_action_distribution.csv")
            self.assertEqual(paths["evaluation_episodes"].name, "double_dqn_evaluation_episodes.csv")
            self.assertEqual(paths["step_trace"].name, "double_dqn_step_trace.csv")
            self.assertTrue(paths["action_distribution"].exists())
            self.assertFalse((root / "action_distribution.csv").exists())

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

    def test_daily_profile_builder_normalizes_large_counts(self):
        """큰 실제 count를 toy 환경 크기에 맞게 max 기준으로 정규화한다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rental_csv = root / "rentals.csv"
            output_path = root / "daily_profile.json"
            header = (
                "자전거번호,대여일시,대여 대여소번호,대여 대여소명,대여거치대,"
                "반납일시,반납대여소번호,반납대여소명,반납거치대,이용시간(분),"
                "이용거리(M),생년,성별,이용자종류,대여대여소ID,반납대여소ID,자전거구분"
            )
            rows = [header]
            station_counts = (10, 8, 6)
            station_ids = ("ST-1", "ST-2", "ST-3")
            station_names = ("마곡A", "마곡B", "마곡C")
            bike_id = 0
            for station_id, station_name, count in zip(station_ids, station_names, station_counts):
                for _ in range(count):
                    bike_id += 1
                    rows.append(
                        f"SPB-{bike_id},2025-01-01 08:00:00,1,{station_name},0,"
                        f"2025-01-01 09:00:00,1,{station_name},0,60,1000,1990,M,"
                        f"내국인,{station_id},{station_id},일반자전거"
                    )
            rental_csv.write_text("\n".join(rows), encoding="utf-8")

            profile = build_daily_profile_from_csvs(
                rental_paths=[rental_csv],
                output_path=output_path,
                station_keyword="마곡",
                station_count=3,
                encoding="utf-8",
                max_daily_count=5,
                show_progress=False,
            )

            self.assertEqual(profile["daily_demand_counts"]["2025-01-01"][8], [5, 4, 3])
            self.assertEqual(profile["metadata"]["normalization"]["raw_max_count"], 10)
            self.assertEqual(profile["metadata"]["normalization"]["max_daily_count"], 5)

    def test_daily_profile_env_uses_real_date_counts(self):
        """daily profile을 읽은 환경이 선택 날짜의 실제 count를 step에 적용한다."""
        config = EnvConfig(
            station_names=("A", "B"),
            demand_ranges={hour: ((0, 0), (0, 0)) for hour in range(24)},
            return_ranges={hour: ((0, 0), (0, 0)) for hour in range(24)},
            daily_dates=("2025-01-01",),
            daily_demand_counts=tuple(
                tuple((2, 1) if hour == 0 else (0, 0) for hour in range(24))
                for _ in range(1)
            ),
            daily_return_counts=tuple(
                tuple((0, 1) if hour == 0 else (0, 0) for hour in range(24))
                for _ in range(1)
            ),
            initial_stock_min=5,
            initial_stock_max=5,
        )
        env = DdareungiEnv(config=config, seed=123)
        env.reset(seed=123)

        _, reward, _, _, info = env.step(0)

        self.assertEqual(info["active_date"], "2025-01-01")
        self.assertEqual(info["demand"], [2, 1])
        self.assertEqual(info["returns"], [0, 1])
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
