"""Episode log를 pygame 창에서 compact game-style로 replay한다."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from ddareungi_rl.stations import STATION_NAMES
from ddareungi_rl.visualization.pixel_replay import load_episode_log, step_records


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 860
TILE_SIZE = 220
TILE_RADIUS = 12
MOVE_FRAMES = 18
HOLD_FRAMES = 16
PANEL_X = 890
PANEL_Y = 150
PANEL_WIDTH = 330
PANEL_HEIGHT = 510
TIMELINE_X = 80
TIMELINE_Y = 760
TIMELINE_WIDTH = 1120
TIMELINE_HEIGHT = 26
CLOSE_BUTTON_X = 1128
CLOSE_BUTTON_Y = 34
CLOSE_BUTTON_WIDTH = 92
CLOSE_BUTTON_HEIGHT = 34

BACKGROUND = (229, 238, 235)
CITY_BLOCK = (239, 244, 241)
ROAD = (153, 162, 160)
ROAD_MARK = (225, 232, 229)
TEXT = (31, 42, 46)
MUTED = (91, 103, 106)
BLUE = (42, 111, 214)
GREEN = (67, 158, 98)
LIGHT_GREEN = (205, 232, 214)
YELLOW = (231, 181, 67)
ORANGE = (223, 132, 58)
RED = (214, 75, 75)
SOFT_RED = (246, 217, 215)
WHITE = (255, 255, 255)
PANEL = (250, 251, 249)
PANEL_BORDER = (199, 210, 207)
TRACK = (218, 226, 222)
DARK = (43, 52, 59)


def require_pygame() -> Any:
    """pygame을 import하고 없으면 설치 안내가 포함된 오류를 발생시킨다."""
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame이 필요합니다. `python -m pip install -e \".[viz]\"`로 설치하세요."
        ) from exc
    return pygame


def station_positions() -> dict[int, tuple[int, int]]:
    """화면에서 삼각형 대여소 tile의 좌상단 좌표를 반환한다."""
    return {
        0: (320, 120),
        1: (100, 455),
        2: (540, 455),
    }


def tile_center(position: tuple[int, int]) -> tuple[int, int]:
    """tile 좌상단 좌표에서 중심 좌표를 계산한다."""
    return (position[0] + TILE_SIZE // 2, position[1] + TILE_SIZE // 2)


def interpolate(
    start: tuple[int, int],
    end: tuple[int, int],
    progress: float,
) -> tuple[int, int]:
    """두 좌표 사이를 progress 비율로 보간한다."""
    x = int(start[0] + (end[0] - start[0]) * progress)
    y = int(start[1] + (end[1] - start[1]) * progress)
    return (x, y)


def stock_color(bikes: int, capacity: int = 10) -> tuple[int, int, int]:
    """재고 수준에 따라 tile 색상을 반환한다."""
    if bikes <= 1:
        return RED
    if bikes <= 3:
        return YELLOW
    if bikes >= capacity - 1:
        return ORANGE
    return GREEN


def station_risk(bikes: int, demand: int = 0) -> tuple[str, tuple[int, int, int]]:
    """대여소 재고와 현재 수요로 부족 위험 label을 계산한다."""
    if bikes <= 0 or demand > bikes:
        return ("헛걸음 위험", RED)
    if bikes <= 2:
        return ("부족 주의", ORANGE)
    return ("안정", GREEN)


def face_mood(bikes: int, demand: int = 0) -> str:
    """대여소 상태를 얼굴 표정 mood로 변환한다."""
    if demand > bikes:
        return "angry"
    if bikes <= 1:
        return "frown"
    if bikes <= 3:
        return "neutral"
    return "smile"


def action_reason(info: dict[str, Any]) -> str:
    """현재 action이 어떤 부족 위험에 대응하는지 짧게 설명한다."""
    action = int(info.get("action", info.get("truck_location", 0)))
    station_bikes = info.get("station_bikes", [0, 0, 0])
    demand = info.get("demand", [0, 0, 0])
    bikes = int(station_bikes[action]) if action < len(station_bikes) else 0
    requested = int(demand[action]) if action < len(demand) else 0

    if int(info.get("unmet_demand", 0)) > 0:
        return "헛걸음 발생 지점 확인"
    if int(info.get("movement_cost", 0)) == 0:
        return "같은 대여소 상태 유지"
    if requested >= bikes:
        return "부족 위험 대여소 대응"
    if bikes <= 2:
        return "낮은 재고 보강"
    return "선택 효과 확인"


def format_rebalance(info: dict[str, Any]) -> str:
    """재배치 이벤트를 발표용 짧은 문장으로 포맷한다."""
    event_type = str(info.get("rebalance_type", "none"))
    amount = int(info.get("rebalance_amount", 0))
    station = int(info.get("rebalance_station", 0))
    names = info.get("station_names", STATION_NAMES)
    station_name = names[station] if isinstance(names, list) and station < len(names) else str(station)

    if amount == 0 or event_type == "none":
        return "재배치 없음"
    if event_type == "load":
        return f"{station_name}에서 자전거 {amount}대 싣기"
    if event_type == "unload":
        return f"{station_name}에 자전거 {amount}대 내리기"
    return f"재배치 {amount}대"


def format_policy_name(info: dict[str, Any]) -> str:
    """log의 policy 이름을 화면용 한국어 label로 변환한다."""
    policy_name = str(info.get("policy_name", "baseline"))
    labels = {
        "random": "무작위 기준 정책",
        "low-stock": "부족 대여소 우선 정책",
        "baseline": "기준 정책",
    }
    return labels.get(policy_name, policy_name)


def format_action(info: dict[str, Any]) -> str:
    """action 번호를 목적지 대여소 이름이 포함된 문장으로 변환한다."""
    action = int(info.get("action", info.get("truck_location", 0)))
    names = info.get("station_names", STATION_NAMES)
    station_name = names[action] if isinstance(names, list) and action < len(names) else str(action)
    return f"{station_name} 방문"


def mission_status(info: dict[str, Any]) -> tuple[str, tuple[int, int, int]]:
    """현재 step의 mission 상태 문구와 색상을 반환한다."""
    unmet_demand = int(info.get("unmet_demand", 0))
    if unmet_demand > 0:
        return (f"미충족 {unmet_demand}건 발생", RED)
    if bool(info.get("service_success", False)):
        return ("수요 방어 성공", GREEN)
    return ("운영 중", BLUE)


def episode_grade(
    cumulative_reward: float,
    cumulative_unmet: int,
    movement_cost: int = 0,
) -> tuple[str, tuple[int, int, int]]:
    """episode 누적 결과를 발표용 등급으로 변환한다."""
    if cumulative_unmet == 0 and cumulative_reward >= 0 and movement_cost <= 8:
        return ("우수", GREEN)
    if cumulative_unmet <= 5 and movement_cost <= 12:
        return ("양호", BLUE)
    if cumulative_unmet <= 15:
        return ("주의", ORANGE)
    return ("개선 필요", RED)


def frame_phase(frame_count: int) -> tuple[float, bool]:
    """현재 animation frame의 이동 progress와 hold 여부를 계산한다."""
    if frame_count < MOVE_FRAMES:
        return frame_count / max(1, MOVE_FRAMES - 1), False
    return 1.0, True


def replay_quit_requested(pygame: Any, event: Any) -> bool:
    """pygame event가 replay 창 종료 요청인지 판정한다."""
    window_close = getattr(pygame, "WINDOWCLOSE", None)
    if event.type == pygame.QUIT:
        return True
    if window_close is not None and event.type == window_close:
        return True
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        return close_button_rect(pygame).collidepoint(event.pos)
    if event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_ESCAPE, pygame.K_q):
            return True
        command_or_ctrl = bool(getattr(event, "mod", 0) & (pygame.KMOD_META | pygame.KMOD_CTRL))
        return command_or_ctrl and event.key in (pygame.K_q, pygame.K_w)
    return False


def replay_quit_key_pressed(pygame: Any) -> bool:
    """현재 키보드 상태에서 replay 종료 키가 눌렸는지 확인한다."""
    pressed = pygame.key.get_pressed()
    return bool(pressed[pygame.K_ESCAPE] or pressed[pygame.K_q])


def close_button_rect(pygame: Any) -> Any:
    """replay 창 안의 클릭 가능한 종료 버튼 영역을 반환한다."""
    return pygame.Rect(
        CLOSE_BUTTON_X,
        CLOSE_BUTTON_Y,
        CLOSE_BUTTON_WIDTH,
        CLOSE_BUTTON_HEIGHT,
    )


def select_font(pygame: Any, size: int, bold: bool = False) -> Any:
    """한국어 표시가 가능한 font를 우선 선택한다."""
    candidates = [
        "AppleGothic",
        "Malgun Gothic",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Arial Unicode MS",
        "Arial",
    ]
    for name in candidates:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    font = pygame.font.Font(None, size)
    font.set_bold(bold)
    return font


def draw_text(
    surface: Any,
    font: Any,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = TEXT,
) -> None:
    """surface에 한 줄 텍스트를 그린다."""
    rendered = font.render(text, True, color)
    surface.blit(rendered, pos)


def draw_rounded_rect(
    pygame: Any,
    surface: Any,
    rect: Any,
    color: tuple[int, int, int],
    border_color: tuple[int, int, int] = WHITE,
) -> None:
    """둥근 사각형 tile을 그리고 외곽선을 추가한다."""
    pygame.draw.rect(surface, color, rect, border_radius=TILE_RADIUS)
    pygame.draw.rect(surface, border_color, rect, width=3, border_radius=TILE_RADIUS)


def draw_small_pill(
    pygame: Any,
    surface: Any,
    fonts: dict[str, Any],
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """짧은 상태 label을 pill 형태로 그린다."""
    rendered = fonts["tiny"].render(text, True, WHITE)
    width = rendered.get_width() + 18
    height = 26
    rect = pygame.Rect(pos[0], pos[1], width, height)
    pygame.draw.rect(surface, color, rect, border_radius=13)
    surface.blit(rendered, (pos[0] + 9, pos[1] + 5))


def draw_bike_icon(
    pygame: Any,
    surface: Any,
    pos: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """작은 자전거 아이콘을 pygame primitive로 그린다."""
    x, y = pos
    pygame.draw.circle(surface, color, (x, y + 10), 7, width=2)
    pygame.draw.circle(surface, color, (x + 28, y + 10), 7, width=2)
    pygame.draw.line(surface, color, (x + 7, y + 10), (x + 14, y), width=2)
    pygame.draw.line(surface, color, (x + 14, y), (x + 21, y + 10), width=2)
    pygame.draw.line(surface, color, (x + 7, y + 10), (x + 21, y + 10), width=2)
    pygame.draw.line(surface, color, (x + 14, y), (x + 18, y - 5), width=2)


def draw_person_icon(
    pygame: Any,
    surface: Any,
    pos: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """수요를 나타내는 작은 사람 아이콘을 그린다."""
    x, y = pos
    pygame.draw.circle(surface, color, (x, y), 5)
    pygame.draw.line(surface, color, (x, y + 6), (x, y + 20), width=3)
    pygame.draw.line(surface, color, (x, y + 11), (x - 7, y + 18), width=2)
    pygame.draw.line(surface, color, (x, y + 11), (x + 7, y + 18), width=2)


def draw_face_icon(
    pygame: Any,
    surface: Any,
    pos: tuple[int, int],
    mood: str,
    color: tuple[int, int, int],
) -> None:
    """대여소 상태를 웃음/무표정/찡그림/화남 얼굴로 그린다."""
    x, y = pos
    pygame.draw.circle(surface, color, (x, y), 20)
    pygame.draw.circle(surface, WHITE, (x, y), 20, width=3)
    pygame.draw.circle(surface, DARK, (x - 7, y - 5), 3)
    pygame.draw.circle(surface, DARK, (x + 7, y - 5), 3)

    if mood == "smile":
        pygame.draw.arc(surface, DARK, (x - 10, y - 4, 20, 18), 3.3, 6.1, width=3)
    elif mood == "neutral":
        pygame.draw.line(surface, DARK, (x - 9, y + 8), (x + 9, y + 8), width=3)
    elif mood == "frown":
        pygame.draw.arc(surface, DARK, (x - 10, y + 5, 20, 18), 0.15, 3.0, width=3)
    else:
        pygame.draw.line(surface, DARK, (x - 12, y - 13), (x - 3, y - 8), width=3)
        pygame.draw.line(surface, DARK, (x + 12, y - 13), (x + 3, y - 8), width=3)
        pygame.draw.arc(surface, DARK, (x - 10, y + 5, 20, 18), 0.15, 3.0, width=3)


def draw_card_shadow(pygame: Any, surface: Any, rect: Any) -> None:
    """카드 뒤에 은은한 그림자를 그린다."""
    shadow = rect.copy()
    shadow.move_ip(0, 5)
    pygame.draw.rect(surface, (202, 213, 209), shadow, border_radius=TILE_RADIUS)


def draw_station_tile(
    pygame: Any,
    surface: Any,
    fonts: dict[str, Any],
    station_id: int,
    info: dict[str, Any],
) -> None:
    """대여소 tile에 이름, 재고, 수요 상태를 간결하게 그린다."""
    positions = station_positions()
    x, y = positions[station_id]
    rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
    station_bikes = info.get("station_bikes", [0, 0, 0])
    demand = info.get("demand", [0, 0, 0])
    returns = info.get("returns", [0, 0, 0])
    names = info.get("station_names", STATION_NAMES)

    bikes = int(station_bikes[station_id])
    requested = int(demand[station_id])
    name = names[station_id] if isinstance(names, list) and station_id < len(names) else str(station_id)
    risk_label, risk_color = station_risk(bikes, requested)
    mood = face_mood(bikes, requested)
    fill = SOFT_RED if risk_color == RED else LIGHT_GREEN if risk_color == GREEN else (250, 237, 205)
    draw_card_shadow(pygame, surface, rect)
    draw_rounded_rect(pygame, surface, rect, fill, border_color=WHITE)
    pygame.draw.rect(surface, risk_color, pygame.Rect(x, y, 10, TILE_SIZE), border_radius=6)
    draw_text(surface, fonts["small"], name, (x + 20, y + 18), TEXT)
    draw_small_pill(pygame, surface, fonts, risk_label, (x + 126, y + 18), risk_color)
    draw_bike_icon(pygame, surface, (x + 26, y + 74), BLUE)
    draw_text(surface, fonts["metric"], f"{bikes:02d}", (x + 82, y + 54), TEXT)
    draw_text(surface, fonts["small"], "대", (x + 156, y + 90), MUTED)
    draw_face_icon(pygame, surface, (x + 42, y + 155), mood, risk_color)
    draw_text(surface, fonts["small"], f"수요 {requested}", (x + 78, y + 136), TEXT)
    draw_text(surface, fonts["small"], f"반납 {returns[station_id]}", (x + 78, y + 162), MUTED)

    if int(info.get("rebalance_station", -1)) == station_id and int(info.get("rebalance_amount", 0)) > 0:
        draw_text(surface, fonts["tiny"], format_rebalance(info), (x + 20, y + 188), BLUE)


def draw_truck_load_hud(
    pygame: Any,
    surface: Any,
    fonts: dict[str, Any],
    info: dict[str, Any],
) -> None:
    """삼각형 중앙에 현재 트럭 적재량만 간결하게 그린다."""
    x, y = 342, 364
    rect = pygame.Rect(x, y, 176, 64)
    draw_card_shadow(pygame, surface, rect)
    draw_rounded_rect(pygame, surface, rect, PANEL, border_color=(210, 218, 226))
    truck_bikes = int(info.get("truck_bikes", 0))
    truck_capacity = int(info.get("truck_capacity", 5))
    draw_text(surface, fonts["tiny"], "트럭 적재", (x + 18, y + 10), MUTED)
    draw_bike_icon(pygame, surface, (x + 18, y + 34), BLUE)
    draw_text(surface, fonts["body"], f"{truck_bikes}/{truck_capacity}", (x + 70, y + 24), BLUE)


def draw_metric_panel(
    pygame: Any,
    surface: Any,
    fonts: dict[str, Any],
    info: dict[str, Any],
) -> None:
    """오른쪽 패널에 하루 목표와 핵심 점수만 그린다."""
    rect = pygame.Rect(PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_HEIGHT)
    reward = float(info.get("reward", 0))
    cumulative_reward = float(info.get("episode_reward_so_far", reward))
    cumulative_color = RED if cumulative_reward < 0 else GREEN
    cumulative_unmet = int(info.get("episode_unmet_demand_so_far", info.get("unmet_demand", 0)))
    cumulative_movement = int(info.get("episode_movement_cost_so_far", info.get("movement_cost", 0)))

    draw_card_shadow(pygame, surface, rect)
    draw_rounded_rect(pygame, surface, rect, PANEL, border_color=PANEL_BORDER)
    draw_text(surface, fonts["small"], "목표", (PANEL_X + 24, PANEL_Y + 26), MUTED)
    draw_text(surface, fonts["body"], "헛걸음 줄이기", (PANEL_X + 24, PANEL_Y + 58), TEXT)
    draw_small_pill(pygame, surface, fonts, str(info.get("learning_stage", "학습 전 기준 정책")), (PANEL_X + 24, PANEL_Y + 104), BLUE)

    draw_text(surface, fonts["small"], "정책", (PANEL_X + 24, PANEL_Y + 154), MUTED)
    draw_text(surface, fonts["small"], format_policy_name(info), (PANEL_X + 24, PANEL_Y + 182), TEXT)

    draw_text(surface, fonts["small"], "운영 점수", (PANEL_X + 24, PANEL_Y + 238), MUTED)
    draw_text(surface, fonts["metric"], f"{cumulative_reward:+.0f}", (PANEL_X + 24, PANEL_Y + 264), cumulative_color)
    draw_text(surface, fonts["small"], f"이번 {reward:+.0f}", (PANEL_X + 164, PANEL_Y + 292), GREEN if reward >= 0 else RED)

    draw_text(surface, fonts["small"], action_reason(info), (PANEL_X + 24, PANEL_Y + 336), BLUE)
    draw_text(surface, fonts["small"], "헛걸음", (PANEL_X + 24, PANEL_Y + 370), MUTED)
    draw_text(surface, fonts["body"], f"{cumulative_unmet}건", (PANEL_X + 24, PANEL_Y + 394), RED if cumulative_unmet else GREEN)
    draw_text(surface, fonts["small"], f"누적 이동 {cumulative_movement}", (PANEL_X + 144, PANEL_Y + 400), MUTED)


def draw_timeline(
    pygame: Any,
    surface: Any,
    fonts: dict[str, Any],
    steps: list[dict[str, Any]],
    step_index: int,
) -> None:
    """하단에 하루 진행률과 실패 지점을 작게 그린다."""
    rect = pygame.Rect(TIMELINE_X, TIMELINE_Y, TIMELINE_WIDTH, TIMELINE_HEIGHT)
    draw_text(
        surface,
        fonts["tiny"],
        "운영 타임라인  초록=정상  빨강=헛걸음  회색=예정",
        (TIMELINE_X, TIMELINE_Y - 24),
        MUTED,
    )
    pygame.draw.rect(surface, WHITE, rect, border_radius=13)
    pygame.draw.rect(surface, PANEL_BORDER, rect, width=2, border_radius=13)

    gap = 4
    usable_width = TIMELINE_WIDTH - 28
    bar_width = max(6, (usable_width - gap * max(0, len(steps) - 1)) // max(1, len(steps)))
    for index, step in enumerate(steps):
        info = step.get("info", {})
        unmet = int(info.get("unmet_demand", 0)) if isinstance(info, dict) else 0
        reward = float(info.get("reward", step.get("reward", 0))) if isinstance(info, dict) else 0.0
        x = TIMELINE_X + 14 + index * (bar_width + gap)
        if index > step_index:
            color = TRACK
        else:
            color = RED if unmet > 0 else GREEN if reward >= 0 else ORANGE
        pygame.draw.rect(
            surface,
            color,
            pygame.Rect(x, TIMELINE_Y + 8, bar_width, TIMELINE_HEIGHT - 16),
            border_radius=4,
        )
        if index == step_index:
            pygame.draw.rect(
                surface,
                DARK,
                pygame.Rect(x - 3, TIMELINE_Y + 5, bar_width + 6, TIMELINE_HEIGHT - 10),
                width=2,
                border_radius=4,
            )


def draw_city_background(pygame: Any, surface: Any) -> None:
    """지도 느낌을 주는 밝은 city block 배경을 그린다."""
    surface.fill(BACKGROUND)
    for rect in [
        pygame.Rect(50, 100, 790, 630),
        pygame.Rect(860, 120, 390, 610),
        pygame.Rect(60, 725, 1160, 102),
    ]:
        pygame.draw.rect(surface, CITY_BLOCK, rect, border_radius=18)


def draw_roads(pygame: Any, surface: Any) -> None:
    """대여소 사이의 단순 도로 선을 그린다."""
    positions = station_positions()
    centers = {key: tile_center(pos) for key, pos in positions.items() if isinstance(key, int)}
    for start, end in [(centers[0], centers[1]), (centers[0], centers[2]), (centers[1], centers[2])]:
        pygame.draw.line(surface, ROAD, start, end, width=12)
        pygame.draw.line(surface, ROAD_MARK, start, end, width=3)


def truck_position(info: dict[str, Any], progress: float) -> tuple[int, int]:
    """이전 위치와 현재 위치 사이에서 트럭 표시 좌표를 계산한다."""
    positions = station_positions()
    previous_location = int(info.get("previous_truck_location", info.get("truck_location", 0)))
    current_location = int(info.get("truck_location", 0))
    start = tile_center(positions[previous_location])
    end = tile_center(positions[current_location])
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(1.0, (dx * dx + dy * dy) ** 0.5)
    inset = min(TILE_SIZE / 2 + 32, distance * 0.45)
    safe_start = (
        int(start[0] + dx / distance * inset),
        int(start[1] + dy / distance * inset),
    )
    safe_end = (
        int(end[0] - dx / distance * inset),
        int(end[1] - dy / distance * inset),
    )
    if previous_location == current_location:
        return (start[0], start[1] + TILE_SIZE // 2 + 18)
    return interpolate(safe_start, safe_end, progress)


def draw_truck(
    pygame: Any,
    surface: Any,
    fonts: dict[str, Any],
    info: dict[str, Any],
    progress: float,
) -> None:
    """현재 animation progress에 맞춰 작은 트럭 아이콘을 그린다."""
    x, y = truck_position(info, progress)
    if y > 540:
        y += 18
    truck_rect = pygame.Rect(x - 23, y - 15, 46, 30)
    pygame.draw.rect(surface, BLUE, truck_rect, border_radius=8)
    pygame.draw.rect(surface, WHITE, truck_rect, width=2, border_radius=8)
    pygame.draw.circle(surface, TEXT, (x - 14, y + 17), 5)
    pygame.draw.circle(surface, TEXT, (x + 14, y + 17), 5)
    draw_text(surface, fonts["tiny"], "트럭", (x - 15, y - 7), WHITE)


def draw_hud(
    surface: Any,
    fonts: dict[str, Any],
    info: dict[str, Any],
    paused: bool,
    hold_phase: bool,
) -> None:
    """상단 scoreboard와 조작 안내를 그린다."""
    phase_text = "싣기/내리기" if hold_phase else "이동 중"
    pause_text = "일시정지" if paused else "재생"
    reward = float(info.get("episode_reward_so_far", info.get("reward", 0)))
    unmet = int(info.get("episode_unmet_demand_so_far", info.get("unmet_demand", 0)))
    draw_text(surface, fonts["title"], "따릉이 재배치 작전", (58, 36), TEXT)
    draw_text(surface, fonts["small"], f"점수 {reward:+.0f}", (450, 48), GREEN if reward >= 0 else RED)
    draw_text(surface, fonts["small"], f"헛걸음 {unmet}", (610, 48), RED if unmet else GREEN)
    draw_text(surface, fonts["small"], f"{int(info.get('time_step', 0)):02d}/24", (770, 48), MUTED)
    draw_text(surface, fonts["small"], format_action(info), (58, 90), MUTED)
    draw_text(surface, fonts["small"], phase_text, (315, 90), ORANGE if hold_phase else BLUE)
    draw_text(surface, fonts["small"], pause_text, (470, 90), RED if paused else GREEN)
    draw_text(
        surface,
        fonts["tiny"],
        "창 클릭 후 Space 일시정지   Right 다음   R 다시보기   Q/Esc 종료",
        (80, 830),
        MUTED,
    )


def draw_close_button(pygame: Any, surface: Any, fonts: dict[str, Any]) -> None:
    """키보드 포커스 없이도 종료할 수 있는 버튼을 그린다."""
    button = close_button_rect(pygame)
    pygame.draw.rect(surface, DARK, button, border_radius=10)
    pygame.draw.rect(surface, WHITE, button, width=2, border_radius=10)
    draw_text(surface, fonts["tiny"], "창 닫기", (button.x + 18, button.y + 9), WHITE)


def draw_episode_summary(
    pygame: Any,
    surface: Any,
    fonts: dict[str, Any],
    info: dict[str, Any],
) -> None:
    """episode 종료 결과를 게임 클리어 화면처럼 표시한다."""
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((20, 28, 34, 170))
    surface.blit(overlay, (0, 0))

    cumulative_reward = float(info.get("episode_reward_so_far", info.get("reward", 0)))
    cumulative_unmet = int(info.get("episode_unmet_demand_so_far", info.get("unmet_demand", 0)))
    movement_cost = int(info.get("episode_movement_cost_so_far", info.get("movement_cost", 0)))
    grade, grade_color = episode_grade(cumulative_reward, cumulative_unmet, movement_cost)
    card = pygame.Rect(430, 270, 420, 280)
    draw_rounded_rect(pygame, surface, card, WHITE, border_color=PANEL_BORDER)
    draw_text(surface, fonts["body"], "하루 재배치 결과", (card.x + 104, card.y + 34), TEXT)
    grade_x = card.x + 92 if len(grade) > 3 else card.x + 126
    draw_text(surface, fonts["metric"], grade, (grade_x, card.y + 78), grade_color)
    draw_text(surface, fonts["small"], f"헛걸음 {cumulative_unmet}건", (card.x + 112, card.y + 150), RED if cumulative_unmet else GREEN)
    draw_text(surface, fonts["small"], f"이동비용 {movement_cost} / 점수 {cumulative_reward:+.0f}", (card.x + 112, card.y + 182), TEXT)
    draw_text(surface, fonts["tiny"], "창 클릭 후 R: 다시 보기   Q/Esc: 종료", (card.x + 84, card.y + 226), MUTED)


def draw_frame(
    pygame: Any,
    surface: Any,
    fonts: dict[str, Any],
    steps: list[dict[str, Any]],
    step_index: int,
    record: dict[str, Any],
    progress: float,
    paused: bool,
    hold_phase: bool,
    ended: bool = False,
) -> None:
    """step record 하나를 pygame surface에 그린다."""
    info = record.get("info", {})
    if not isinstance(info, dict):
        raise ValueError("step record must contain an info dict")

    draw_city_background(pygame, surface)
    draw_roads(pygame, surface)
    for station_id in range(3):
        draw_station_tile(pygame, surface, fonts, station_id, info)
    draw_truck(pygame, surface, fonts, info, progress)
    draw_truck_load_hud(pygame, surface, fonts, info)
    draw_metric_panel(pygame, surface, fonts, info)
    draw_timeline(pygame, surface, fonts, steps, step_index)
    draw_hud(surface, fonts, info, paused, hold_phase)
    if ended:
        draw_episode_summary(pygame, surface, fonts, info)
    draw_close_button(pygame, surface, fonts)


def replay_window(
    records: list[dict[str, Any]],
    fps: int = 30,
    max_steps: int | None = None,
    loop: bool = False,
) -> None:
    """episode step record를 pygame 창에서 순서대로 재생한다."""
    pygame = require_pygame()
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Ddareungi FrozenLake-style Replay")
    clock = pygame.time.Clock()
    fonts = {
        "title": select_font(pygame, 28, bold=True),
        "body": select_font(pygame, 23),
        "small": select_font(pygame, 18),
        "tiny": select_font(pygame, 15),
        "metric": select_font(pygame, 46, bold=True),
    }

    steps = step_records(records)
    if max_steps is not None:
        steps = steps[:max_steps]
    if not steps:
        pygame.display.quit()
        pygame.quit()
        return

    step_index = 0
    frame_count = 0
    paused = False
    ended = False
    running = True
    hold_summary = os.environ.get("SDL_VIDEODRIVER") != "dummy"

    try:
        while running:
            for event in pygame.event.get():
                if replay_quit_requested(pygame, event):
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        step_index = 0
                        frame_count = 0
                        ended = False
                    elif event.key == pygame.K_RIGHT:
                        step_index = min(step_index + 1, len(steps) - 1)
                        frame_count = MOVE_FRAMES
                        ended = False

            if running and replay_quit_key_pressed(pygame):
                running = False
            if not running:
                break

            progress, hold_phase = frame_phase(frame_count)
            draw_frame(
                pygame,
                screen,
                fonts,
                steps,
                step_index,
                steps[step_index],
                progress,
                paused,
                hold_phase,
                ended,
            )
            pygame.display.flip()

            if not paused and not ended:
                frame_count += 1
                if frame_count >= MOVE_FRAMES + HOLD_FRAMES:
                    frame_count = 0
                    step_index += 1
                    if step_index >= len(steps):
                        if loop:
                            step_index = 0
                        else:
                            step_index = len(steps) - 1
                            ended = True
                            if not hold_summary:
                                running = False

            clock.tick(fps)

    finally:
        pygame.display.quit()
        pygame.quit()


def parse_args() -> argparse.Namespace:
    """pygame replay CLI argument를 파싱한다."""
    parser = argparse.ArgumentParser(description="Replay a Ddareungi episode log in a pygame window.")
    parser.add_argument("log_path", type=Path, help="Path to an episode JSON log.")
    parser.add_argument("--fps", type=int, default=30, help="Window replay frame rate.")
    parser.add_argument("--max-steps", type=int, default=None, help="Only replay N steps.")
    parser.add_argument("--loop", action="store_true", help="Loop replay until the user quits.")
    parser.add_argument(
        "--dummy-video",
        action="store_true",
        help="Use SDL dummy video driver for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    """episode log를 읽어 pygame 창 replay를 실행한다."""
    args = parse_args()
    if args.dummy_video:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    records = load_episode_log(args.log_path)
    replay_window(records, fps=args.fps, max_steps=args.max_steps, loop=args.loop)


if __name__ == "__main__":
    main()
