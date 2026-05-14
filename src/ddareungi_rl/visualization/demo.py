"""V0 episode 생성과 pygame replay를 한 번에 실행하는 demo CLI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ddareungi_rl.training.evaluate import evaluate, save_episode_log
from ddareungi_rl.visualization.pixel_replay import load_episode_log
from ddareungi_rl.visualization.pygame_replay import replay_window


def create_demo_log(policy: str, seed: int, log_path: Path) -> Path:
    """baseline episode 하나를 실행하고 demo replay log를 저장한다."""
    results = evaluate(
        policy_name=policy,  # type: ignore[arg-type]
        episodes=1,
        seed=seed,
        render_mode="none",
    )
    save_episode_log(results[0], log_path)
    return log_path


def parse_args() -> argparse.Namespace:
    """demo CLI argument를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="V0 episode 하나를 생성하고 pygame 창에서 바로 replay한다."
    )
    parser.add_argument(
        "--policy",
        choices=["random", "low-stock"],
        default="low-stock",
        help="demo episode 생성에 사용할 baseline policy.",
    )
    parser.add_argument("--seed", type=int, default=42, help="episode 생성 seed.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/demo_episode.json"),
        help="생성된 episode log를 저장할 경로.",
    )
    parser.add_argument("--fps", type=int, default=30, help="창 replay frame rate.")
    parser.add_argument("--max-steps", type=int, default=None, help="처음 N step만 replay한다.")
    parser.add_argument("--loop", action="store_true", help="사용자가 종료할 때까지 replay를 반복한다.")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="pygame 창을 열지 않고 episode log만 생성한다.",
    )
    parser.add_argument(
        "--dummy-video",
        action="store_true",
        help="smoke test용 SDL dummy video driver를 사용한다.",
    )
    return parser.parse_args()


def main() -> None:
    """demo episode log를 만들고 선택적으로 pygame replay 창을 연다."""
    args = parse_args()
    log_path = create_demo_log(args.policy, args.seed, args.log_path)
    print(f"demo log saved: {log_path}")

    if args.no_window:
        return

    if args.dummy_video:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    records = load_episode_log(log_path)
    replay_window(records, fps=args.fps, max_steps=args.max_steps, loop=args.loop)


if __name__ == "__main__":
    main()
