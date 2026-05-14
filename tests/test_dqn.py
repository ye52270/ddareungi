import tempfile
from pathlib import Path
import unittest

from ddareungi_rl.agents import DQNAgent, DQNConfig, DQNPolicy, QNetwork, ReplayBuffer
from ddareungi_rl.agents.dqn import Transition
from ddareungi_rl.envs import ToyDdareungiEnv


class DQNTest(unittest.TestCase):
    """순수 Python DQN 구성 요소의 smoke test 모음."""

    def test_q_network_predicts_one_value_per_action(self):
        """QNetwork가 action 개수만큼 Q-value를 반환하는지 검증한다."""
        network = QNetwork(input_size=6, hidden_size=8, output_size=3, seed=123)

        q_values = network.predict([0.1, 0.2, 0.3, 0.0, 0.5, 0.1])

        self.assertEqual(len(q_values), 3)

    def test_replay_buffer_samples_transitions(self):
        """ReplayBuffer가 transition을 저장하고 sample을 반환하는지 검증한다."""
        buffer = ReplayBuffer(capacity=3, seed=123)
        transition = Transition([0.0], 0, -1.0, [0.1], False)

        buffer.add(transition)

        self.assertEqual(buffer.sample(1), [transition])

    def test_dqn_agent_selects_valid_action(self):
        """DQNAgent가 action space 안의 greedy action을 반환하는지 검증한다."""
        agent = DQNAgent(6, 3, DQNConfig(hidden_size=8), seed=123)

        action = agent.select_greedy_action([0.1, 0.2, 0.3, 0.0, 0.5, 0.1])

        self.assertIn(action, {0, 1, 2})

    def test_dqn_agent_save_load_roundtrip(self):
        """DQNAgent를 저장한 뒤 DQNPolicy로 다시 불러올 수 있는지 검증한다."""
        env = ToyDdareungiEnv(seed=123)
        env.reset(seed=123)
        agent = DQNAgent(env.observation_size, env.action_space_n, seed=123)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "dqn.json"
            agent.save(model_path)
            policy = DQNPolicy(model_path)

            action = policy.select_action(env)

        self.assertIn(action, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
