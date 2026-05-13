import unittest

from ddareungi_rl.envs import ToyDdareungiEnv
from ddareungi_rl.policies import LowStockPolicy


class ToyDdareungiEnvTest(unittest.TestCase):
    """Toy 따릉이 환경의 smoke test 모음."""

    def test_env_runs_full_episode_with_low_stock_policy(self):
        """단순 heuristic이 full episode를 끝까지 실행할 수 있는지 검증한다."""
        env = ToyDdareungiEnv(seed=123)
        policy = LowStockPolicy()
        state, info = env.reset(seed=123)

        self.assertEqual(len(state), env.observation_size)
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
        self.assertIsInstance(total_reward, float)

    def test_ansi_render_returns_text_frame(self):
        """ANSI 렌더링이 주요 label을 포함한 text frame을 반환하는지 검증한다."""
        env = ToyDdareungiEnv(render_mode="ansi", seed=123)
        env.reset(seed=123)

        frame = env.render()

        self.assertIsInstance(frame, str)
        self.assertIn("Toy Ddareungi V0", frame)
        self.assertIn("truck_loc", frame)


if __name__ == "__main__":
    unittest.main()
