import unittest

from ai_trainer.game_ui.analysis_log import SessionAnalysisLog
from ai_trainer.rep_detector_2d import Adaptive2DRepDetector, validate_dtw_candidate


class SemanticRepPipelineTests(unittest.TestCase):
    def detector(self):
        return Adaptive2DRepDetector(46.38858745051192, 27.433186772026716)

    def run_sequence(self, values, velocities=None):
        detector = self.detector()
        events = []
        velocities = velocities or [0.0] * len(values)
        for frame, ((knee, hip), velocity) in enumerate(zip(values, velocities)):
            event = detector.update(
                frame, knee, hip, pelvis_velocity=velocity, pelvis_near_baseline=frame > len(values) - 4
            )
            if event:
                events.append(event)
        return events

    def normal_motion(self):
        return [
            (0, 0), (6, 4), (12, 8), (25, 16), (45, 27), (60, 35),
            (52, 31), (42, 25), (30, 18), (15, 8), (7, 4), (4, 2), (2, 1),
        ]

    def test_1_standing_squat_standing_is_one_rep(self):
        self.assertEqual(self.run_sequence(self.normal_motion()).count("rep_end"), 1)

    def test_2_no_bottom_pause_is_one_rep(self):
        values = [(0, 0), (12, 8), (28, 17), (55, 32), (44, 25), (25, 14), (8, 4), (4, 2), (1, 1), (0, 0)]
        self.assertEqual(self.run_sequence(values).count("rep_end"), 1)

    def test_3_standing_jitter_is_zero_rep(self):
        values = [(2, 1), (-1, 2), (3, -1), (1, 2), (0, 0), (2, 1)]
        self.assertNotIn("rep_end", self.run_sequence(values))

    def test_4_partial_descent_is_not_rep(self):
        values = [(0, 0), (10, 6), (13, 8), (11, 6), (7, 4), (2, 1), (0, 0)]
        self.assertNotIn("rep_end", self.run_sequence(values))

    def test_5_3d_pelvis_jitter_does_not_hide_clear_2d_rep(self):
        velocities = [0.03, -0.04, 0.02, -0.03, 0.04, -0.02, 0.03, -0.04, 0.02, -0.03, 0.04, -0.02, 0.01]
        self.assertEqual(self.run_sequence(self.normal_motion(), velocities).count("rep_end"), 1)

    def test_6_hip_down_plus_2d_insufficient_stays_hip_down(self):
        self.assertEqual(
            validate_dtw_candidate("엉덩이하방오류", depth_2d_pass=False, consistency_pass=True),
            "엉덩이하방오류",
        )

    def test_7_sufficient_2d_and_unreliable_3d_blocks_hip_down(self):
        self.assertEqual(
            validate_dtw_candidate("엉덩이하방오류", depth_2d_pass=True, consistency_pass=False),
            "자세추정불확실",
        )

    def test_8_rejected_hip_down_never_becomes_normal(self):
        self.assertNotEqual(
            validate_dtw_candidate("엉덩이하방오류", depth_2d_pass=True, consistency_pass=True),
            "정상",
        )

    def test_9_three_completed_reps_create_three_log_entries(self):
        from types import SimpleNamespace

        history = SessionAnalysisLog()
        for index in range(3):
            result = SimpleNamespace(
                rep_index=index,
                predicted_class="자세추정불확실",
                score_vs_normal=None,
                raw_distance_by_class={"엉덩이하방오류": 3.0},
                top_contributing_features=[],
            )
            history.record_completed_rep("에어스쿼트", result)
        self.assertEqual(len(history.entries), 3)
        self.assertEqual(history.summary("에어스쿼트").total_reps, 3)


if __name__ == "__main__":
    unittest.main()
