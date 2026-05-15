import unittest

from ddareungi_rl.visualization.pygame_replay import (
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    action_reason,
    close_button_rect,
    draw_timeline,
    draw_truck_load_hud,
    episode_grade,
    face_mood,
    format_action,
    format_policy_name,
    format_rebalance,
    frame_phase,
    interpolate,
    mission_status,
    replay_quit_key_pressed,
    replay_quit_requested,
    replay_window,
    require_pygame,
    select_font,
    station_positions,
    station_risk,
    stock_color,
    truck_position,
)


class PygameReplayTest(unittest.TestCase):
    """pygame 창 replay helper의 smoke test 모음."""

    def test_interpolate_moves_between_points(self):
        """좌표 보간 함수가 시작점과 끝점 사이 값을 반환하는지 검증한다."""
        self.assertEqual(interpolate((0, 0), (10, 20), 0.5), (5, 10))

    def test_frame_phase_enters_hold_after_move_frames(self):
        """이동 frame 이후 hold phase로 전환되는지 검증한다."""
        progress, hold_phase = frame_phase(999)

        self.assertEqual(progress, 1.0)
        self.assertTrue(hold_phase)

    def test_format_rebalance_uses_korean_station_name(self):
        """재배치 이벤트 문구에 한국식 대여소 이름이 들어가는지 검증한다."""
        text = format_rebalance(
            {
                "rebalance_type": "unload",
                "rebalance_amount": 2,
                "rebalance_station": 0,
                "station_names": ["마포구청역", "여의도역", "서울숲입구"],
            }
        )

        self.assertIn("마포구청역", text)
        self.assertIn("내리기", text)

    def test_stock_color_marks_low_stock_as_red(self):
        """낮은 재고가 빨간색 계열로 표시되는지 검증한다."""
        self.assertEqual(stock_color(0), (214, 75, 75))

    def test_format_action_uses_station_name(self):
        """action 설명에 목적지 대여소 이름이 들어가는지 검증한다."""
        text = format_action(
            {
                "action": 2,
                "station_names": ["마포구청역", "여의도역", "서울숲입구"],
            }
        )

        self.assertIn("서울숲입구", text)
        self.assertIn("방문", text)

    def test_format_policy_name_labels_low_stock(self):
        """low-stock policy 이름을 화면용 label로 변환하는지 검증한다."""
        self.assertEqual(
            format_policy_name({"policy_name": "low-stock"}),
            "현재 재고 최저 정책",
        )

    def test_station_risk_marks_shortage(self):
        """수요가 재고보다 큰 대여소를 헛걸음 위험으로 표시하는지 검증한다."""
        label, _ = station_risk(bikes=1, demand=3)

        self.assertEqual(label, "헛걸음 위험")

    def test_face_mood_marks_angry_when_demand_exceeds_stock(self):
        """수요가 재고보다 크면 화난 얼굴 mood를 반환하는지 검증한다."""
        self.assertEqual(face_mood(bikes=1, demand=3), "angry")

    def test_action_reason_describes_step_outcome(self):
        """action 설명이 선택 이유가 아니라 step 결과를 반환하는지 검증한다."""
        reason = action_reason(
            {
                "action": 1,
                "station_bikes": [5, 1, 6],
                "demand": [1, 2, 1],
                "unmet_demand": 2,
                "movement_cost": 1,
            }
        )

        self.assertIn("선택 결과", reason)
        self.assertIn("헛걸음", reason)

    def test_action_reason_marks_same_location_action(self):
        """같은 대여소를 다시 선택하면 현 위치 유지로 설명하는지 검증한다."""
        reason = action_reason({"same_location_action": True, "movement_cost": 0})

        self.assertIn("현 위치 유지", reason)

    def test_mission_status_marks_unmet_as_failure(self):
        """미충족 수요가 있는 step을 실패 신호로 표시하는지 검증한다."""
        text, _ = mission_status({"unmet_demand": 2})

        self.assertIn("미충족", text)

    def test_episode_grade_marks_good_episode_as_clear(self):
        """누적 미충족이 낮은 episode를 clear 등급으로 표시하는지 검증한다."""
        grade, _ = episode_grade(cumulative_reward=-5, cumulative_unmet=3)

        self.assertEqual(grade, "양호")

    def test_episode_grade_penalizes_large_movement_cost(self):
        """이동비용이 큰 episode를 과하게 좋은 등급으로 표시하지 않는지 검증한다."""
        grade, _ = episode_grade(
            cumulative_reward=1,
            cumulative_unmet=0,
            movement_cost=20,
        )

        self.assertEqual(grade, "주의")

    def test_truck_position_stays_on_route_between_tiles(self):
        """트럭 표시 위치가 대여소 card 중심이 아니라 도로 쪽에 머무는지 검증한다."""
        pos = truck_position(
            {
                "previous_truck_location": 0,
                "truck_location": 2,
            },
            1.0,
        )

        self.assertLess(pos[1], station_positions()[2][1])

    def test_station_positions_only_include_stations(self):
        """pygame 지도 좌표가 실제 action 대상인 대여소만 포함하는지 검증한다."""
        self.assertEqual(set(station_positions()), {0, 1, 2})

    def test_replay_quit_requested_handles_escape_q_and_close(self):
        """Esc, Q, 창 닫기, 종료 버튼 click이 replay 종료 요청으로 판정되는지 검증한다."""
        pygame = require_pygame()
        pygame.init()

        escape_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
        q_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q)
        close_event = pygame.event.Event(pygame.QUIT)
        button = close_button_rect(pygame)
        button_event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            button=1,
            pos=button.center,
        )
        space_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)

        self.assertTrue(replay_quit_requested(pygame, escape_event))
        self.assertTrue(replay_quit_requested(pygame, q_event))
        self.assertTrue(replay_quit_requested(pygame, close_event))
        self.assertTrue(replay_quit_requested(pygame, button_event))
        self.assertFalse(replay_quit_requested(pygame, space_event))

        pygame.quit()

    def test_replay_quit_requested_handles_command_shortcuts(self):
        """macOS/desktop 닫기 단축키가 replay 종료 요청으로 판정되는지 검증한다."""
        pygame = require_pygame()
        pygame.init()

        command_q = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_q,
            mod=pygame.KMOD_META,
        )
        command_w = pygame.event.Event(
            pygame.KEYDOWN,
            key=pygame.K_w,
            mod=pygame.KMOD_META,
        )

        self.assertTrue(replay_quit_requested(pygame, command_q))
        self.assertTrue(replay_quit_requested(pygame, command_w))

        pygame.quit()

    def test_replay_quit_key_pressed_defaults_to_false(self):
        """키 입력이 없을 때 pressed-key 종료 보강이 False를 반환하는지 검증한다."""
        import os

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame = require_pygame()
        pygame.init()
        pygame.display.set_mode((1, 1))

        self.assertFalse(replay_quit_key_pressed(pygame))

        pygame.display.quit()
        pygame.quit()

    def test_draw_truck_load_hud_runs_without_extra_action_tile(self):
        """중앙 트럭 적재 HUD가 dummy surface에서 렌더링되는지 검증한다."""
        import os

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame = require_pygame()
        pygame.init()
        pygame.font.init()
        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        fonts = {
            "body": select_font(pygame, 22),
            "tiny": select_font(pygame, 14),
        }

        draw_truck_load_hud(
            pygame,
            surface,
            fonts,
            {"truck_bikes": 3, "truck_capacity": 5, "unmet_demand": 0},
        )

        pygame.quit()

    def test_replay_window_runs_with_dummy_video(self):
        """dummy video driver에서 1-step 창 replay가 종료되는지 검증한다."""
        records = [
            {
                "event": "step",
                "action": 1,
                "reward": 0,
                "info": {
                    "station_names": ["마포구청역", "여의도역", "서울숲입구"],
                    "time_step": 1,
                    "previous_truck_location": 0,
                    "truck_location": 1,
                    "truck_bikes": 0,
                    "station_bikes": [6, 5, 2],
                    "action": 1,
                    "demand": [1, 0, 0],
                    "returns": [0, 0, 0],
                    "unmet_demand": 0,
                    "full_returns": 0,
                    "movement_cost": 1,
                    "reward": 0,
                    "rebalance_type": "unload",
                    "rebalance_amount": 3,
                    "rebalance_station": 1,
                },
            }
        ]

        import os

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        replay_window(records, fps=120, max_steps=1)

    def test_draw_timeline_runs_with_future_steps_hidden(self):
        """timeline이 미래 step을 포함해도 렌더링이 깨지지 않는지 검증한다."""
        import os

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame = require_pygame()
        pygame.init()
        pygame.font.init()
        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        fonts = {"tiny": select_font(pygame, 14)}
        steps = [
            {"info": {"reward": 1, "unmet_demand": 0}},
            {"info": {"reward": -10, "unmet_demand": 1}},
            {"info": {"reward": 1, "unmet_demand": 0}},
        ]

        draw_timeline(pygame, surface, fonts, steps, step_index=1)

        pygame.quit()


if __name__ == "__main__":
    unittest.main()
