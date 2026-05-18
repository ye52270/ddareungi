import unittest

from ddareungi_rl.baselines import LowStockPolicy, NoOpPolicy, RandomPolicy
from ddareungi_rl.dqn import DQNConfig, evaluate_policy, train_dqn
from ddareungi_rl.env import DdareungiEnv, EnvConfig


class SimpleProjectTest(unittest.TestCase):
    """단순화된 프로젝트의 핵심 흐름을 검증한다."""

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
        self.assertIn("reward_formula", info)

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


if __name__ == "__main__":
    unittest.main()
