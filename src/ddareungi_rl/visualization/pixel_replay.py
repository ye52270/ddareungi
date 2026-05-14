"""Episode log를 FrozenLake 스타일 text tile로 replay한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
GRAY = "\033[90m"


def colorize(text: str, color: str, enabled: bool) -> str:
    """색상 사용 여부에 따라 ANSI color code를 적용한다."""
    if not enabled:
        return text
    return f"{color}{text}{RESET}"


def load_episode_log(path: Path) -> list[dict[str, Any]]:
    """JSON 파일에서 episode replay log를 읽어온다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("episode log must be a list of records")
    return data


def step_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """전체 log에서 step record만 순서대로 추출한다."""
    return [record for record in records if record.get("event") == "step"]


def bike_color(bikes: int, enabled: bool) -> str:
    """자전거 재고 수준에 따라 표시 색상을 고른다."""
    if bikes <= 1:
        return colorize(f"{bikes:02d}", RED, enabled)
    if bikes <= 3:
        return colorize(f"{bikes:02d}", YELLOW, enabled)
    return colorize(f"{bikes:02d}", GREEN, enabled)


def reward_color(reward: float, enabled: bool) -> str:
    """reward 값이 좋고 나쁨에 따라 표시 색상을 고른다."""
    if reward < 0:
        return colorize(f"{reward:.0f}", RED, enabled)
    if reward > 0:
        return colorize(f"{reward:.0f}", GREEN, enabled)
    return colorize(f"{reward:.0f}", YELLOW, enabled)


def padded_color(text: str, width: int, color_code: str, enabled: bool) -> str:
    """문자열 폭을 먼저 맞춘 뒤 ANSI 색상을 적용한다."""
    padded = f"{text:<{width}}"
    return colorize(padded, color_code, enabled)


def station_tile(
    name: str,
    station_id: int,
    bikes: int,
    truck_location: int,
    color: bool,
) -> list[str]:
    """대여소 하나를 고정 폭 tile 문자열 목록으로 만든다."""
    has_truck = station_id == truck_location
    marker = colorize("T", BLUE, color) if has_truck else " "
    label = padded_color(name, 10, BOLD, color)
    stock = bike_color(bikes, color)
    return [
        f" {label} {marker} ",
        f" bikes {stock}     ",
    ]


def render_frame(record: dict[str, Any], color: bool = True) -> str:
    """step record 하나를 2x2 tile replay frame으로 렌더링한다."""
    info = record.get("info", {})
    if not isinstance(info, dict):
        raise ValueError("step record must contain an info dict")

    station_bikes = info.get("station_bikes", [0, 0, 0])
    truck_location = int(info.get("truck_location", 0))
    truck_bikes = int(info.get("truck_bikes", 0))
    time_step = int(info.get("time_step", 0))
    action = info.get("action", record.get("action", "-"))
    demand = info.get("demand", "-")
    returns = info.get("returns", "-")
    unmet = int(info.get("unmet_demand", 0))
    full_returns = int(info.get("full_returns", 0))
    movement_cost = int(info.get("movement_cost", 0))
    reward = float(info.get("reward", record.get("reward", 0)))

    home = station_tile("HOME", 0, int(station_bikes[0]), truck_location, color)
    work = station_tile("WORK", 1, int(station_bikes[1]), truck_location, color)
    park = station_tile("PARK", 2, int(station_bikes[2]), truck_location, color)
    load_text = padded_color(str(truck_bikes), 7, BLUE, color)
    depot = [
        f" {padded_color('TRUCK', 10, BOLD, color)}   ",
        f" load {load_text} ",
    ]

    title = colorize("Ddareungi Tile Replay", MAGENTA + BOLD, color)
    unmet_text = colorize(str(unmet), RED if unmet else GREEN, color)
    full_text = colorize(str(full_returns), YELLOW if full_returns else GREEN, color)
    reward_text = reward_color(reward, color)
    dim_line = colorize("-" * 35, GRAY, color)

    return "\n".join(
        [
            f"{title}",
            dim_line,
            f"+--------------+--------------+",
            f"|{home[0]}|{work[0]}|",
            f"|{home[1]}|{work[1]}|",
            f"+--------------+--------------+",
            f"|{park[0]}|{depot[0]}|",
            f"|{park[1]}|{depot[1]}|",
            f"+--------------+--------------+",
            f"time={time_step:02d}/24  action={action}  reward={reward_text}",
            f"unmet={unmet_text}  full_returns={full_text}  move_cost={movement_cost}",
            f"demand={demand}",
            f"returns={returns}",
        ]
    )


def replay_frames(
    records: list[dict[str, Any]],
    max_steps: int | None = None,
    color: bool = True,
) -> list[str]:
    """episode log에서 replay frame 목록을 만든다."""
    steps = step_records(records)
    if max_steps is not None:
        steps = steps[:max_steps]
    return [render_frame(record, color=color) for record in steps]


def parse_args() -> argparse.Namespace:
    """pixel replay CLI argument를 파싱한다."""
    parser = argparse.ArgumentParser(description="Replay a Ddareungi episode log as tiles.")
    parser.add_argument("log_path", type=Path, help="Path to an episode JSON log.")
    parser.add_argument("--max-steps", type=int, default=None, help="Only replay N steps.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between frames.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the terminal between frames for animation-like playback.",
    )
    return parser.parse_args()


def main() -> None:
    """episode log를 읽어 tile replay frame을 출력한다."""
    args = parse_args()
    records = load_episode_log(args.log_path)
    frames = replay_frames(
        records,
        max_steps=args.max_steps,
        color=not args.no_color,
    )

    for index, frame in enumerate(frames):
        if args.clear:
            print("\033[2J\033[H", end="")
        print(frame)
        if index < len(frames) - 1:
            print()
            if args.delay > 0:
                time.sleep(args.delay)


if __name__ == "__main__":
    main()
