import json
from pathlib import Path
import tempfile
import unittest

from ddareungi_rl.agents import DQNConfig
from ddareungi_rl.training.evaluate import aggregate_action_counts, evaluate, same_location_rate
from ddareungi_rl.training.train_dqn import save_metrics, train_dqn


class TrainDQNTest(unittest.TestCase):
    """DQN 학습 루프와 평가 통합의 smoke test 모음."""

    def test_train_dqn_saves_model_and_metrics(self):
        """짧은 DQN 학습 결과를 model/metrics 파일로 저장할 수 있는지 검증한다."""
        config = DQNConfig(
            hidden_size=8,
            batch_size=4,
            min_replay_size=4,
            target_update_interval=2,
            epsilon_decay_steps=10,
        )
        agent, metrics, last_result = train_dqn(episodes=2, seed=123, config=config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "dqn.json"
            metrics_path = tmp_path / "metrics.json"
            agent.save(model_path)
            save_metrics(metrics, metrics_path)

            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(len(metrics), 2)
        self.assertEqual(last_result.steps, 24)
        self.assertTrue(saved_metrics)

    def test_evaluate_runs_with_saved_dqn_model(self):
        """저장된 DQN model을 evaluate.py의 dqn policy로 평가할 수 있는지 검증한다."""
        config = DQNConfig(
            hidden_size=8,
            batch_size=4,
            min_replay_size=4,
            target_update_interval=2,
            epsilon_decay_steps=10,
        )
        agent, _, _ = train_dqn(episodes=1, seed=123, config=config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "dqn.json"
            agent.save(model_path)
            results = evaluate(
                policy_name="dqn",
                episodes=1,
                seed=456,
                render_mode="none",
                model_path=model_path,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].steps, 24)
        self.assertEqual(sum(aggregate_action_counts(results).values()), 24)
        self.assertGreaterEqual(same_location_rate(results), 0.0)


if __name__ == "__main__":
    unittest.main()
