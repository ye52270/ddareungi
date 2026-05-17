import unittest

import gymnasium as gym

from ddareungi_rl.envs import ToyDdareungiEnv
from ddareungi_rl.policies import DemandAwarePolicy, LowStockPolicy


class ToyDdareungiEnvTest(unittest.TestCase):
    """Toy 따릉이 환경의 smoke test 모음."""

    def test_env_runs_full_episode_with_low_stock_policy(self):
        """단순 heuristic이 full episode를 끝까지 실행할 수 있는지 검증한다."""
        env = ToyDdareungiEnv(seed=123)
        policy = LowStockPolicy()
        state, info = env.reset(seed=123)

        self.assertIsInstance(env, gym.Env)
        self.assertEqual(len(state), env.observation_size)
        self.assertTrue(env.observation_space.contains(state))
        self.assertEqual(env.action_space.n, env.action_space_n)
        self.assertEqual(info["time_step"], 0)

        done = False
        steps = 0
        total_reward = 0.0
        while not done:
            action = policy.select_action(env)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        self.assertEqual(steps, env.config.episode_steps)
        self.assertEqual(len(state), env.observation_size)
        self.assertTrue(env.observation_space.contains(state))
        self.assertIsInstance(total_reward, float)

    def test_ansi_render_returns_text_frame(self):
        """ANSI 렌더링이 주요 label을 포함한 text frame을 반환하는지 검증한다."""
        env = ToyDdareungiEnv(render_mode="ansi", seed=123)
        env.reset(seed=123)

        frame = env.render()

        self.assertIsInstance(frame, str)
        self.assertIn("Toy Ddareungi V0", frame)
        self.assertIn("truck_loc", frame)

    def test_step_info_contains_animation_fields(self):
        """step info에 replay animation용 필드가 포함되는지 검증한다."""
        env = ToyDdareungiEnv(seed=123)
        env.reset(seed=123)

        _, _, _, _, info = env.step(1)

        self.assertIn("station_names", info)
        self.assertIn("truck_previous_location", info)
        self.assertIn("rebalance_type", info)
        self.assertIn("rebalance_amount", info)
        self.assertIn("station_bikes_before_rebalance", info)
        self.assertIn("station_bikes_after_rebalance", info)
        self.assertIn("station_bikes_after_rebalance_all", info)
        self.assertIn("station_bikes_after_demand", info)
        self.assertIn("decision_station_bikes", info)
        self.assertIn("same_location_action", info)
        self.assertIn("service_success", info)
        self.assertEqual(info["decision_station_bikes"], info["previous_station_bikes"])
        self.assertEqual(info["decision_truck_location"], info["previous_truck_location"])

    def test_reward_uses_unmet_demand_and_movement_cost(self):
        """V0 reward가 미충족 수요와 이동비용만 반영하는지 검증한다."""
        env = ToyDdareungiEnv(seed=123)
        env.reset(seed=123)

        _, reward, _, _, info = env.step(0)

        expected_reward = -10 * int(info["unmet_demand"]) - int(info["movement_cost"])
        self.assertEqual(reward, expected_reward)

    def test_demand_aware_policy_uses_expected_shortage(self):
        """예상 수요 부족 baseline이 시간대별 수요와 재고 차이를 사용함을 검증한다."""
        env = ToyDdareungiEnv(seed=123)
        env.reset(seed=123)
        env.time_step = 7
        env.station_bikes = [2, 4, 4]
        policy = DemandAwarePolicy()

        action = policy.select_action(env)

        self.assertEqual(action, 0)


if __name__ == "__main__":
    unittest.main()
