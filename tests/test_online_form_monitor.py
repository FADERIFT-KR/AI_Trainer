import unittest
from unittest.mock import patch

import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.heel_contact import estimate_sequence_heel_contact
from ai_trainer.online_dtw import OnlineSquatSession
from ai_trainer.reference_levels import DIFFICULTY_LEVELS, REFERENCE_CLASSES
from ai_trainer.reference_matching import INDETERMINATE_CLASS


_INDEX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}


def _monitor_session() -> OnlineSquatSession:
    """Build only the state required by the frame-local monitoring helpers."""
    session = OnlineSquatSession.__new__(OnlineSquatSession)
    session.ema_alpha = 0.5
    session.ema_raw2d_prev = None
    session.state = "prep"
    session.calib_frames = 2
    session.scale2d = 100.0
    session.heel_lift_threshold = 0.06
    session.prep_ground_samples = []
    session.prep_torso_samples = []
    session.prep_hip_samples = []
    session.ground_y = None
    session.baseline_ankle_ground_heights = None
    session.baseline_toe_ground_heights = None
    session.baseline_heel_ground_heights = None
    session.baseline_torso_inclination = None
    session.baseline_hip_flexion = None
    session.heel_lift_streak = np.zeros(2, dtype=np.int32)
    session.heel_lift_delta = np.zeros(2, dtype=np.float64)
    session.bottom_entry_torso = None
    session.bottom_entry_hip = None
    session.previous_torso_inclination = None
    session.previous_hip_flexion = None
    session.butt_wink_latched = False
    session.butt_wink_bottom_delta_deg = 8.0
    session.butt_wink_rate_deg = 4.0
    return session


def _feet_at(y: float) -> np.ndarray:
    points = np.zeros((len(COMMON_JOINT_NAMES), 2), dtype=np.float64)
    for name in ("LBigToe", "RBigToe", "LAnkle", "RAnkle", "LHeel", "RHeel"):
        points[_INDEX[name], 1] = y
    return points


class OnlineFormMonitorTests(unittest.TestCase):
    def test_offline_heel_proxy_uses_preparation_phase_baseline(self):
        frames = [_feet_at(400.0) for _ in range(8)]
        for frame in frames:
            frame[_INDEX["Hip"], 1] = 300.0
            frame[_INDEX["Neck"], 1] = 200.0
        frames[3][_INDEX["LHeel"], 1] = 389.0
        frames[4][_INDEX["LHeel"], 1] = 389.0
        contact, delta = estimate_sequence_heel_contact(
            np.stack(frames),
            [0, 2],
            ema_alpha=1.0,
        )
        self.assertIs(contact, False)
        self.assertGreater(delta[0], 0.10)

    def test_ema_is_causal_and_short(self):
        session = _monitor_session()
        first = np.zeros((len(COMMON_JOINT_NAMES), 2), dtype=np.float64)
        second = np.full_like(first, 10.0)
        np.testing.assert_array_equal(session._ema_filter_2d(first), first)
        np.testing.assert_array_equal(session._ema_filter_2d(second), np.full_like(first, 5.0))

    def test_heel_lift_uses_preparation_heel_and_two_frames(self):
        session = _monitor_session()
        coords = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
        prep = _feet_at(400.0)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        self.assertEqual(session.ground_y, 400.0)

        session.state = "descend"
        lifted = _feet_at(400.0)
        lifted[_INDEX["LHeel"], 1] = 390.0
        lifted[_INDEX["RHeel"], 1] = 390.0
        self.assertEqual(session._form_warnings(coords, lifted, 10.0, 170.0, None), ())
        warnings = session._form_warnings(coords, lifted, 10.0, 170.0, None)
        self.assertIn("HEEL LIFT (L/R)", warnings)
        np.testing.assert_allclose(session.heel_lift_delta, (0.1, 0.1))

    def test_ankle_motion_without_heel_motion_does_not_warn(self):
        session = _monitor_session()
        coords = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
        prep = _feet_at(400.0)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)

        session.state = "descend"
        moved_ankles = _feet_at(400.0)
        moved_ankles[_INDEX["LAnkle"], 1] = 380.0
        moved_ankles[_INDEX["RAnkle"], 1] = 380.0
        self.assertEqual(
            session._form_warnings(coords, moved_ankles, 10.0, 170.0, None), ()
        )
        np.testing.assert_allclose(session.heel_lift_delta, (0.0, 0.0))

    def test_completed_rep_reports_stable_heel_contact(self):
        session = _monitor_session()
        coords = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
        prep = _feet_at(400.0)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        session.raw2d_buffer = [_feet_at(400.0) for _ in range(6)]

        contact, max_delta = session._heel_contact_evidence(0, 5)

        self.assertIs(contact, True)
        np.testing.assert_allclose(max_delta, (0.0, 0.0))

    def test_completed_rep_detects_persistent_heel_lift_with_grounded_toe(self):
        session = _monitor_session()
        coords = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
        prep = _feet_at(400.0)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        frames = [_feet_at(400.0) for _ in range(6)]
        for frame in frames[2:4]:
            frame[_INDEX["LHeel"], 1] = 390.0
        session.raw2d_buffer = frames

        contact, max_delta = session._heel_contact_evidence(0, 5)

        self.assertIs(contact, False)
        np.testing.assert_allclose(max_delta, (0.1, 0.0))

    def test_completed_rep_with_moving_toes_has_insufficient_evidence(self):
        session = _monitor_session()
        coords = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
        prep = _feet_at(400.0)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        frames = [_feet_at(400.0) for _ in range(6)]
        for frame in frames[1:]:
            frame[_INDEX["LBigToe"], 1] = 380.0
            frame[_INDEX["RBigToe"], 1] = 380.0
        session.raw2d_buffer = frames

        contact, _ = session._heel_contact_evidence(0, 5)

        self.assertIsNone(contact)

    def test_stable_2d_heels_suppress_final_dtw_heel_error(self):
        session = _monitor_session()
        coords = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
        prep = _feet_at(400.0)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        session._capture_preparation_baseline(coords, prep, 0.0, 180.0, True)
        session.raw2d_buffer = [_feet_at(400.0) for _ in range(6)]
        session.rep_start_idx = 0
        session.emit_offset = 0
        session.aligned_seq = [coords.copy() for _ in range(6)]
        session.phase_boundaries_running = {}
        session.db_operational = {
            level: {class_label: [{}] for class_label in REFERENCE_CLASSES}
            for level in DIFFICULTY_LEVELS
        }
        session.weights_cfg = {"rejection": {}}
        session.weight_profile = "test"
        session.score_calib = None
        session.algorithm = "dtw"
        session.completed_reps = []

        comparison = {
            "min_distance": 1.0,
            "best_detail": {"per_feature_contrib": {"heel_height": 1.0}},
        }
        with (
            patch("ai_trainer.online_dtw.extract_all_features", return_value={}),
            patch("ai_trainer.online_dtw.resolve_weights", return_value={}),
            patch("ai_trainer.online_dtw.multi_reference_distance", return_value=comparison),
        ):
            session._finalize_rep(5)

        result = session.completed_reps[-1]
        self.assertEqual(result.predicted_class, INDETERMINATE_CLASS)
        self.assertEqual(result.decision_status, "indeterminate")
        self.assertEqual(result.rejection_reason, "heel_error_not_supported_by_2d")
        self.assertIs(result.heel_contact_2d, True)

    def test_bottom_pelvic_tuck_is_reported_as_a_risk_proxy(self):
        session = _monitor_session()
        session.state = "bottom"
        session.baseline_torso_inclination = 0.0
        session.baseline_hip_flexion = 180.0
        coords = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
        image = _feet_at(400.0)
        self.assertEqual(session._form_warnings(coords, image, 12.0, 165.0, None), ())
        warnings = session._form_warnings(coords, image, 22.0, 155.0, None)
        self.assertIn("BUTT WINK RISK", warnings)


if __name__ == "__main__":
    unittest.main()
