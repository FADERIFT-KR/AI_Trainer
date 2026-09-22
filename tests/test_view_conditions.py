"""Headless tests for view-specific conditions and 3x3 decision fusion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.squat.camera_views import VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT  # noqa: E402
from ai_trainer.core.common_skeleton import COMMON_JOINT_NAMES  # noqa: E402
from ai_trainer.squat.game_ui.framing_check import check_framing, upright_calibration_pose  # noqa: E402
from ai_trainer.squat.session_decision import UNCERTAIN_CLASS, decide_session, decide_view  # noqa: E402
from ai_trainer.squat.view_conditions import (  # noqa: E402
    assess_paper_squat_conditions,
    assess_view_rep,
    extract_view_features,
    load_view_condition_config,
    summarize_view_features,
)


_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def squat_sequence(n_frames: int = 8) -> np.ndarray:
    """Return a simple, non-degenerate 2D common skeleton sequence."""
    base = {
        "LShoulder": (-0.35, 2.8), "RShoulder": (0.35, 2.8),
        "LElbow": (-0.55, 2.2), "RElbow": (0.55, 2.2),
        "LWrist": (-0.60, 1.7), "RWrist": (0.60, 1.7),
        "LHip": (-0.25, 1.8), "RHip": (0.25, 1.8),
        "LKnee": (-0.32, 0.9), "RKnee": (0.32, 0.9),
        "LAnkle": (-0.38, 0.0), "RAnkle": (0.38, 0.0),
        "LHeel": (-0.45, 0.0), "RHeel": (0.31, 0.0),
        "LBigToe": (-0.18, 0.0), "RBigToe": (0.58, 0.0),
        "Hip": (0.0, 1.8), "Neck": (0.0, 2.8),
    }
    result = np.zeros((n_frames, len(COMMON_JOINT_NAMES), 2), dtype=np.float64)
    for frame in range(n_frames):
        depth = 0.45 * np.sin(np.pi * frame / max(n_frames - 1, 1))
        for name, point in base.items():
            result[frame, _IDX[name]] = point
        for name in ("LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "Hip", "Neck"):
            result[frame, _IDX[name], 1] -= depth
    return result


class ViewFeatureTests(unittest.TestCase):
    def test_features_are_invariant_to_uniform_image_scale_and_translation(self) -> None:
        coords = squat_sequence()
        transformed = coords * 173.0 + np.array([512.0, 288.0])
        for view in (VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT):
            original = extract_view_features(coords, view)
            changed = extract_view_features(transformed, view)
            self.assertEqual(original.keys(), changed.keys())
            for key in original:
                np.testing.assert_allclose(original[key], changed[key], atol=1e-6, rtol=1e-6)

    def test_phase_summary_and_rule_assessment_are_auditable(self) -> None:
        coords = squat_sequence()
        bounds = {"준비": [0, 2], "하강": [2, 4], "최저점": [4, 5], "상승": [5, 8]}
        summary = summarize_view_features(extract_view_features(coords, VIEW_FRONT), bounds)
        metric = "준비|stance_width_ratio|median"
        self.assertIn(metric, summary)
        threshold = summary[metric] - 0.01
        config = {
            "normal_ranges": {VIEW_FRONT: {}},
            "error_rules": {
                VIEW_FRONT: {
                    "테스트오류": [{
                        "metric": metric,
                        "operator": ">=",
                        "threshold": threshold,
                        "validation_balanced_accuracy": 0.9,
                    }]
                }
            },
        }
        assessed = assess_view_rep(coords, bounds, VIEW_FRONT, config)
        self.assertEqual(assessed.predicted_error, "테스트오류")
        self.assertEqual(assessed.error_support["테스트오류"], 1.0)
        self.assertEqual(len(assessed.rule_hits), 1)

        config["responsible_views_by_error"] = {"테스트오류": [VIEW_LEFT]}
        excluded = assess_view_rep(coords, bounds, VIEW_FRONT, config)
        self.assertIsNone(excluded.predicted_error)
        self.assertEqual(excluded.error_support["테스트오류"], 0.0)

    def test_generated_config_keeps_fit_tune_and_validation_actors_disjoint(self) -> None:
        config = load_view_condition_config()
        self.assertEqual(config["schema_version"], 2)
        split = config["rule_selection_actor_split"]
        fit, tune, validation = map(set, (split["fit"], split["tune"], split["validation"]))
        self.assertTrue(fit)
        self.assertTrue(tune)
        self.assertTrue(validation)
        self.assertFalse(fit & tune)
        self.assertFalse(fit & validation)
        self.assertFalse(tune & validation)
        for views in config["responsible_views_by_error"].values():
            self.assertTrue({VIEW_LEFT, VIEW_RIGHT} <= set(views))
            self.assertTrue(set(views) <= {VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT})

    def test_paper_bottom_conditions_are_side_only_and_report_actionable_violation(self) -> None:
        coords = squat_sequence()
        # A right-side lowest position with a horizontal thigh and aligned toe,
        # but a nearly closed knee-hip-shoulder angle.  This isolates paper
        # condition 1 from conditions 2 and 3.
        for frame in range(3, 5):
            coords[frame, _IDX["RHip"]] = (0.0, 1.0)
            coords[frame, _IDX["RKnee"]] = (0.8, 1.0)
            coords[frame, _IDX["RAnkle"]] = (0.8, 2.0)
            coords[frame, _IDX["RBigToe"]] = (0.75, 2.0)
            coords[frame, _IDX["RShoulder"]] = (0.8, 1.0)
        bounds = {"최저점": [3, 5]}

        violations = assess_paper_squat_conditions(coords, bounds, VIEW_RIGHT)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].condition, "조건 1 · 고관절 각도")
        self.assertIn("엉덩이를 더 높게", violations[0].message)
        self.assertEqual(assess_paper_squat_conditions(coords, bounds, VIEW_FRONT), ())

        assessed = assess_view_rep(coords, bounds, VIEW_RIGHT, {
            "normal_ranges": {VIEW_RIGHT: {}}, "error_rules": {VIEW_RIGHT: {}},
        })
        self.assertEqual(assessed.to_dict()["paper_posture_violations"][0]["condition"],
                         "조건 1 · 고관절 각도")


def framing_landmarks(side: bool) -> np.ndarray:
    landmarks = np.zeros((33, 4), dtype=np.float64)
    landmarks[:, 3] = 1.0
    points = {
        0: (0.50, 0.10),
        11: (0.40, 0.25), 12: (0.60, 0.25),
        23: (0.43, 0.50), 24: (0.57, 0.50),
        25: (0.43, 0.70), 26: (0.57, 0.70),
        27: (0.43, 0.89), 28: (0.57, 0.89),
        29: (0.42, 0.91), 30: (0.58, 0.91),
    }
    if side:
        points.update({
            11: (0.49, 0.25), 12: (0.51, 0.25),
            23: (0.495, 0.50), 24: (0.505, 0.50),
            25: (0.49, 0.70), 26: (0.51, 0.70),
            27: (0.49, 0.89), 28: (0.51, 0.89),
            29: (0.47, 0.91), 30: (0.53, 0.91),
        })
    for index, (x, y) in points.items():
        landmarks[index, :2] = (x, y)
    return landmarks


class FramingViewTests(unittest.TestCase):
    def test_upright_calibration_rejects_a_bent_or_leaning_pose(self) -> None:
        side = framing_landmarks(side=True)
        self.assertTrue(upright_calibration_pose(side, VIEW_RIGHT).ok)

        bent = side.copy()
        bent[24, :2] = (0.505, 0.50)
        bent[26, :2] = (0.60, 0.62)
        bent[28, :2] = (0.51, 0.79)
        result = upright_calibration_pose(bent, VIEW_RIGHT)
        self.assertFalse(result.ok)
        self.assertIn("무릎을 펴고", result.message)

        leaning = side.copy()
        leaning[12, 0] = 0.70
        result = upright_calibration_pose(leaning, VIEW_RIGHT)
        self.assertFalse(result.ok)
        self.assertIn("상체를 세우고", result.message)

    def test_dataset_calibrated_gate_distinguishes_front_and_side(self) -> None:
        front = framing_landmarks(side=False)
        side = framing_landmarks(side=True)
        self.assertTrue(check_framing(front, 1280, 720, view=VIEW_FRONT).ok)
        self.assertFalse(check_framing(side, 1280, 720, view=VIEW_FRONT).ok)
        self.assertTrue(check_framing(side, 1280, 720, view=VIEW_LEFT).ok)
        self.assertTrue(check_framing(side, 1280, 720, view=VIEW_RIGHT).ok)
        self.assertFalse(check_framing(front, 1280, 720, view=VIEW_LEFT).ok)

    def test_side_squat_does_not_lose_tracking_at_bottom(self) -> None:
        # Saved right-side session: hip/leg screen ratio rises at the bottom
        # even though the subject has not rotated toward the camera.
        bent = framing_landmarks(side=True)
        bent[23, :2] = (0.46, 0.53)
        bent[24, :2] = (0.54, 0.53)
        bent[25, :2] = (0.59, 0.62)
        bent[26, :2] = (0.60, 0.62)
        bent[27, :2] = (0.49, 0.79)
        bent[28, :2] = (0.51, 0.79)
        self.assertFalse(check_framing(bent, 1280, 720, view=VIEW_RIGHT).ok)
        self.assertTrue(check_framing(bent, 1280, 720, relax_distance=True, view=VIEW_RIGHT).ok)
        bent[28, 3] = 0.1
        self.assertFalse(check_framing(bent, 1280, 720, relax_distance=True, view=VIEW_RIGHT).ok)


def rep(view: str, model_class: str, error: str | None = None, support: float = 0.0,
        paper_violation: bool = False) -> dict:
    error_support = {error: support} if error else {}
    return {
        "view_mode": view,
        "model_class": model_class,
        "condition_assessment": {
            "predicted_error": error,
            "error_support": error_support,
            "paper_posture_violations": ([{"condition": "조건 1 · 고관절 각도"}]
                                         if paper_violation else []),
        },
    }


class SessionDecisionTests(unittest.TestCase):
    def test_one_rep_cannot_count_twice_toward_two_of_three_confirmation(self) -> None:
        repetitions = [
            rep(VIEW_FRONT, "발뒤꿈치오류", "발뒤꿈치오류", 0.9),
            rep(VIEW_FRONT, "정상"),
            rep(VIEW_FRONT, "정상"),
        ]
        decision = decide_view(repetitions, VIEW_FRONT)
        self.assertEqual(decision.label, "정상")
        self.assertEqual(decision.confirmed_errors, ())

    def test_repeated_error_in_one_responsible_view_controls_session_result(self) -> None:
        repetitions = []
        repetitions.extend([rep(VIEW_FRONT, "정상") for _ in range(3)])
        repetitions.extend([
            rep(VIEW_LEFT, "발뒤꿈치오류", "발뒤꿈치오류", 0.9),
            rep(VIEW_LEFT, "발뒤꿈치오류", "발뒤꿈치오류", 0.8),
            rep(VIEW_LEFT, "정상"),
        ])
        repetitions.extend([rep(VIEW_RIGHT, "정상") for _ in range(3)])
        decision = decide_session(repetitions)
        self.assertEqual(decision.label, "발뒤꿈치오류")
        self.assertEqual(decision.confirmed_errors, ("발뒤꿈치오류",))

    def test_repeated_rule_only_evidence_vetoes_normal_but_cannot_declare_error(self) -> None:
        repetitions = [
            rep(VIEW_FRONT, "정상", "발뒤꿈치오류", 0.9),
            rep(VIEW_FRONT, "정상", "발뒤꿈치오류", 0.8),
            rep(VIEW_FRONT, "정상"),
        ]
        decision = decide_view(repetitions, VIEW_FRONT)
        self.assertEqual(decision.label, UNCERTAIN_CLASS)
        self.assertEqual(decision.confirmed_errors, ())

    def test_missing_view_remains_uncertain_instead_of_becoming_normal(self) -> None:
        repetitions = [rep(VIEW_FRONT, "정상") for _ in range(3)]
        decision = decide_session(repetitions)
        self.assertEqual(decision.label, UNCERTAIN_CLASS)

    def test_repeated_paper_posture_violation_is_final_error(self) -> None:
        repetitions = []
        repetitions.extend([rep(VIEW_FRONT, "정상") for _ in range(3)])
        repetitions.extend([
            rep(VIEW_LEFT, "정상", paper_violation=True),
            rep(VIEW_LEFT, "정상", paper_violation=True),
            rep(VIEW_LEFT, "정상"),
        ])
        repetitions.extend([rep(VIEW_RIGHT, "정상") for _ in range(3)])
        decision = decide_session(repetitions)
        self.assertEqual(decision.label, "논문 자세 기준 위반")


if __name__ == "__main__":
    unittest.main()
