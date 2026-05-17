import json
from pathlib import Path
import tempfile
import unittest

from ddareungi_rl.agents.torch_dqn import TorchDQNConfig, torch
from ddareungi_rl.training.evaluate import aggregate_action_counts, evaluate
from ddareungi_rl.training.train_torch_dqn import save_torch_metrics, train_torch_dqn
from tests.profile_fixture import write_tiny_profile


@unittest.skipIf(torch is None, "PyTorch가 설치된 환경에서만 실행한다.")
class TrainTorchDQNTest(unittest.TestCase):
    """PyTorch DQN 학습 루프와 평가 통합의 smoke test 모음."""

    def test_train_torch_dqn_saves_model_and_metrics(self):
        """짧은 PyTorch DQN 학습 결과를 model/metrics 파일로 저장할 수 있는지 검증한다."""
        config = TorchDQNConfig(
            hidden_size=8,
            batch_size=4,
            min_replay_size=4,
            target_update_interval=2,
            epsilon_decay_steps=10,
        )
        agent, metrics, last_result = train_torch_dqn(episodes=2, seed=123, config=config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "torch_dqn.pt"
            metrics_path = tmp_path / "metrics.json"
            agent.save(model_path)
            save_torch_metrics(metrics, metrics_path)

            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(len(metrics), 2)
        self.assertEqual(last_result.steps, 24)
        self.assertTrue(saved_metrics)

    def test_evaluate_runs_with_saved_torch_dqn_model(self):
        """저장된 PyTorch DQN model을 evaluate.py의 torch-dqn policy로 평가할 수 있는지 검증한다."""
        config = TorchDQNConfig(
            hidden_size=8,
            batch_size=4,
            min_replay_size=4,
            target_update_interval=2,
            epsilon_decay_steps=10,
        )
        agent, _, _ = train_torch_dqn(episodes=1, seed=123, config=config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "torch_dqn.pt"
            agent.save(model_path)
            results = evaluate(
                policy_name="torch-dqn",
                episodes=1,
                seed=456,
                render_mode="none",
                model_path=model_path,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].steps, 24)
        self.assertEqual(sum(aggregate_action_counts(results).values()), 24)

    def test_train_and_evaluate_with_profile_path(self):
        """real-profile JSON을 주입한 PyTorch DQN 학습/평가 smoke test."""
        config = TorchDQNConfig(
            hidden_size=8,
            batch_size=4,
            min_replay_size=4,
            target_update_interval=2,
            epsilon_decay_steps=10,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            profile_path = write_tiny_profile(tmp_path / "profile.json")
            model_path = tmp_path / "torch_dqn.pt"
            agent, metrics, last_result = train_torch_dqn(
                episodes=1,
                seed=123,
                config=config,
                profile_path=profile_path,
            )
            agent.save(model_path)
            results = evaluate(
                policy_name="torch-dqn",
                episodes=1,
                seed=456,
                render_mode="none",
                model_path=model_path,
                profile_path=profile_path,
            )

        self.assertEqual(len(metrics), 1)
        self.assertEqual(last_result.steps, 24)
        self.assertEqual(results[0].steps, 24)


if __name__ == "__main__":
    unittest.main()
