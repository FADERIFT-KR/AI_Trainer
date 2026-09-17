import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from ai_trainer.game_ui.analysis_log import SessionAnalysisLog
from ai_trainer.game_ui.cv_text import display_text, safe_put_text
from ai_trainer.game_ui.framing_check import check_framing
from ai_trainer.rep_detector_2d import validate_dtw_candidate


class _Status(Enum):
    READY = "위치 좋음"


class _Result:
    message = "전신이 화면에 들어오지 않았습니다"


class CvTextAndQualityTests(unittest.TestCase):
    def test_1_normal_string_is_accepted_by_put_text(self):
        frame = np.zeros((40, 120, 3), dtype=np.uint8)
        cv2.putText(frame, display_text("위치 좋음"), (2, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        self.assertEqual(display_text("위치 좋음"), "위치 좋음")

    def test_2_none_becomes_empty_string_without_crash(self):
        self.assertEqual(display_text(None), "")

    def test_3_enum_and_message_object_use_display_fields(self):
        self.assertEqual(display_text(_Status.READY), "위치 좋음")
        self.assertEqual(display_text(_Result()), "전신이 화면에 들어오지 않았습니다")

    def test_4_missing_full_body_blocks_framing(self):
        landmarks = np.zeros((33, 4), dtype=np.float32)
        landmarks[:, :2] = .5
        landmarks[:, 3] = 1.0
        landmarks[27, 3] = 0.0
        result = check_framing(landmarks, 640, 480)
        self.assertFalse(result.ok)
        self.assertIn("발", result.message)

    def test_5_position_warning_messages_draw_without_type_error(self):
        frame = np.zeros((80, 320, 3), dtype=np.uint8)
        for source, message in (
            ("too_close", SimpleNamespace(message="카메라에서 조금 멀어져 주세요")),
            ("feet_not_visible", SimpleNamespace(message="발이 화면에 보이도록 이동해 주세요")),
        ):
            self.assertIs(
                safe_put_text(frame, message, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, source=source),
                frame,
            )

    def test_6_consistency_failure_stays_unknown(self):
        self.assertEqual(
            validate_dtw_candidate("엉덩이하방오류", depth_2d_pass=True, consistency_pass=False),
            "자세추정불확실",
        )

    def test_7_ui_text_failure_is_isolated(self):
        frame = np.zeros((40, 120, 3), dtype=np.uint8)
        with patch("ai_trainer.game_ui.cv_text.cv2.putText", side_effect=TypeError("bad text")):
            returned = safe_put_text(frame, "warning", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, source="forced_failure")
        self.assertIs(returned, frame)

    def test_8_four_completed_reps_create_four_log_entries(self):
        history = SessionAnalysisLog()
        for index in range(4):
            result = SimpleNamespace(
                rep_index=index,
                predicted_class="자세추정불확실",
                score_vs_normal=None,
                raw_distance_by_class={"엉덩이하방오류": 3.0},
                top_contributing_features=[],
                debug_summary={"reason_code": "UNKNOWN_2D_3D_MISMATCH"},
            )
            history.record_completed_rep("에어스쿼트", result)
        self.assertEqual(history.summary("에어스쿼트").total_reps, 4)

    def test_9_unknown_reason_is_saved_in_analysis_log(self):
        result = SimpleNamespace(
            rep_index=0,
            predicted_class="자세추정불확실",
            score_vs_normal=None,
            raw_distance_by_class={"엉덩이하방오류": 3.0},
            top_contributing_features=[],
            debug_summary={"reason_code": "UNKNOWN_2D_3D_MISMATCH"},
        )
        history = SessionAnalysisLog()
        entry = history.record_completed_rep("에어스쿼트", result)
        self.assertEqual(len(history.entries), 1)
        self.assertIn("2D와 3D", entry.error_messages[0])


if __name__ == "__main__":
    unittest.main()
