import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from ddareungi_rl.cli import (
    run_baseline_suite,
    run_choice,
    run_dqn_small_evaluation,
    run_dqn_small_training,
    replay_log,
    summarize_results,
)


class CLITest(unittest.TestCase):
    """콘솔 메뉴 helper의 smoke test 모음."""

    def test_summarize_results_contains_core_metrics(self):
        """baseline 결과 요약에 핵심 metric이 포함되는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "low_stock_log.json"
            results_by_policy = run_baseline_suite(
                episodes=1,
                seed=123,
                save_low_stock_log=log_path,
            )

            summary = summarize_results("random", results_by_policy["random"])

            self.assertTrue(log_path.exists())

        self.assertIn("avg_reward", summary)
        self.assertIn("avg_unmet_demand", summary)

    def test_dqn_menu_training_and_evaluation_helpers_run(self):
        """DQN 메뉴 helper가 학습 모델을 만들고 평가까지 실행하는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "dqn.json"
            metrics_path = tmp_path / "metrics.json"
            train_log_path = tmp_path / "train_log.json"
            eval_log_path = tmp_path / "eval_log.json"

            run_dqn_small_training(
                episodes=1,
                seed=123,
                model_path=model_path,
                metrics_path=metrics_path,
                log_path=train_log_path,
            )
            results = run_dqn_small_evaluation(
                episodes=1,
                seed=1000,
                model_path=model_path,
                log_path=eval_log_path,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].steps, 24)

    def test_run_choice_exits_on_zero(self):
        """메뉴 선택 0이 반복 종료 신호를 반환하는지 검증한다."""
        self.assertFalse(run_choice("0"))

    def test_visualization_menu_choice_exits_after_replay(self):
        """visualization 메뉴 선택이 replay 이후 메뉴 반복 종료 신호를 반환하는지 검증한다."""
        with patch("ddareungi_rl.cli.run_baseline_suite", return_value={"low-stock": []}), patch(
            "ddareungi_rl.cli.summarize_results",
            return_value="summary",
        ), patch("ddareungi_rl.cli.replay_log"):
            self.assertFalse(run_choice("4"))

    def test_dqn_visualization_menu_choice_exits_after_replay(self):
        """DQN visualization 메뉴 선택도 replay 이후 메뉴 반복 종료 신호를 반환하는지 검증한다."""
        with patch("ddareungi_rl.cli.run_dqn_small_evaluation", return_value=[]), patch(
            "ddareungi_rl.cli.summarize_results",
            return_value="summary",
        ), patch("ddareungi_rl.cli.replay_log"):
            self.assertFalse(run_choice("5"))

    def test_dqn_menu_choice_returns_to_menu_when_model_is_missing(self):
        """DQN 모델이 없으면 메뉴가 예외로 죽지 않고 반복을 계속하는지 검증한다."""
        with patch(
            "ddareungi_rl.cli.run_dqn_small_evaluation",
            side_effect=FileNotFoundError("missing model"),
        ):
            self.assertTrue(run_choice("3"))

    def test_replay_log_runs_with_dummy_video(self):
        """저장된 log를 visualization helper로 replay할 수 있는지 검증한다."""
        import os

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "low_stock_log.json"
            run_baseline_suite(episodes=1, seed=123, save_low_stock_log=log_path)

            replay_log(log_path, max_steps=1)


if __name__ == "__main__":
    unittest.main()
