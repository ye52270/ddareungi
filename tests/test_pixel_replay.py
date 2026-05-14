import unittest

from ddareungi_rl.visualization.pixel_replay import render_frame, replay_frames


class PixelReplayTest(unittest.TestCase):
    """Episode log tile replay의 smoke test 모음."""

    def test_render_frame_contains_core_fields(self):
        """단일 step record가 핵심 지표를 포함한 frame으로 렌더링되는지 검증한다."""
        record = {
            "event": "step",
            "action": 2,
            "reward": -30,
            "info": {
                "time_step": 7,
                "truck_location": 2,
                "truck_bikes": 0,
                "station_bikes": [2, 10, 1],
                "action": 2,
                "demand": [4, 1, 3],
                "returns": [1, 4, 1],
                "unmet_demand": 3,
                "full_returns": 0,
                "movement_cost": 1,
                "reward": -30,
            },
        }

        frame = render_frame(record, color=False)

        self.assertIn("Ddareungi Tile Replay", frame)
        self.assertIn("HOME", frame)
        self.assertIn("WORK", frame)
        self.assertIn("PARK", frame)
        self.assertIn("time=07/24", frame)
        self.assertIn("unmet=3", frame)

    def test_replay_frames_uses_only_step_records(self):
        """reset record를 건너뛰고 step record만 frame으로 변환하는지 검증한다."""
        records = [
            {"event": "reset", "state": [], "info": {}},
            {
                "event": "step",
                "action": 0,
                "reward": 1,
                "info": {
                    "time_step": 1,
                    "truck_location": 0,
                    "truck_bikes": 2,
                    "station_bikes": [5, 4, 3],
                    "action": 0,
                    "demand": [0, 1, 0],
                    "returns": [1, 0, 0],
                    "unmet_demand": 0,
                    "full_returns": 0,
                    "movement_cost": 0,
                    "reward": 1,
                },
            },
        ]

        frames = replay_frames(records, color=False)

        self.assertEqual(len(frames), 1)
        self.assertIn("reward=1", frames[0])


if __name__ == "__main__":
    unittest.main()
