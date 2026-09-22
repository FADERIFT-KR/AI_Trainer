from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.squat.game_ui.post_session_replay import annotate_replay_frame, build_replay_views, error_joints


class PostSessionReplayTests(unittest.TestCase):
    def _write_view(self, directory: Path) -> None:
        (directory / "front.json").write_text(json.dumps({
            "view": "front", "video_file": "front.mp4", "timeline_file": "front.jsonl",
            "video_fps": 30.0,
        }), encoding="utf-8")
        points = np.column_stack((np.arange(18) * 10 + 30, np.full(18, 50))).tolist()
        rows = []
        for frame in range(12):
            rows.append({"video_frame": frame, "analysis_frame": frame - 2,
                         "common_2d": points,
                         "completed_rep": {"frame_range": [2, 6]} if frame == 8 else None})
        (directory / "front.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )

    def test_final_error_maps_only_its_analysis_range_to_video_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._write_view(directory)
            views = build_replay_views(directory, [{
                "view_mode": "front", "view_rep_index": 0, "sequence_class": "무릎 오류",
                "display_class": "무릎 오류", "fail_reason": "무릎 각도 범위 초과",
            }])
        self.assertEqual(len(views), 1)
        span = views[0].spans[0]
        self.assertEqual((span.start_video_frame, span.end_video_frame), (4, 8))
        self.assertIn("LKnee", span.joints)
        self.assertIn("RKnee", span.joints)

    def test_normal_rep_is_not_replayed_as_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self._write_view(directory)
            views = build_replay_views(directory, [{
                "view_mode": "front", "view_rep_index": 0, "sequence_class": "정상",
                "display_class": "정상",
            }])
        self.assertEqual(views, [])

    def test_annotation_changes_only_error_span_and_uses_red_joint_marker(self):
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        points = np.column_stack((np.arange(18) * 10 + 20, np.full(18, 60))).tolist()
        span = type("Span", (), {"start_video_frame": 2, "end_video_frame": 4,
                                  "joints": ("LKnee",), "label": "무릎 오류", "message": "검증된 오류"})()
        untouched = annotate_replay_frame(frame, {"video_frame": 1, "common_2d": points}, [span])
        marked = annotate_replay_frame(frame, {"video_frame": 3, "common_2d": points}, [span])
        self.assertEqual(int(np.count_nonzero(untouched)), 0)
        self.assertGreater(int(np.count_nonzero(marked[:, :, 2])), 0)
        self.assertIn("Hip", error_joints("고관절 오류"))
        self.assertEqual(len(COMMON_JOINT_NAMES), 18)


if __name__ == "__main__":
    unittest.main()
