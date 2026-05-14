import json
from pathlib import Path
import tempfile
import unittest

from ddareungi_rl.visualization.demo import create_demo_log


class DemoTest(unittest.TestCase):
    """한 줄 demo wrapper의 smoke test 모음."""

    def test_create_demo_log_writes_episode_json(self):
        """demo log 생성 함수가 step record를 포함한 JSON 파일을 쓰는지 검증한다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "demo.json"

            create_demo_log("low-stock", 42, log_path)

            records = json.loads(log_path.read_text(encoding="utf-8"))
            step_records = [record for record in records if record.get("event") == "step"]
            self.assertTrue(step_records)
            info = step_records[0]["info"]
            self.assertIn("station_names", info)
            self.assertIn("truck_location", info)
            self.assertIn("station_bikes", info)
            self.assertIn("policy_name", info)
            self.assertIn("episode_reward_so_far", info)
            self.assertIn("episode_served_demand_so_far", info)
            self.assertIn("episode_unmet_demand_so_far", info)
            self.assertIn("episode_total_demand_so_far", info)
            self.assertIn("service_rate_so_far", info)


if __name__ == "__main__":
    unittest.main()
