import tempfile
import unittest

import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.features import extract_all_features
from ai_trainer.posture_score import COMPONENTS, PostureScorer, RealtimePostureMatcher
from ai_trainer.two_d_diagnostic import extract_2d_features


I = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}


def sequence(amplitude=1.0, frames=24):
    p = np.zeros((frames, 18, 3), dtype=float)
    phase = np.sin(np.linspace(0.0, np.pi, frames)) * amplitude
    for t, bend in enumerate(phase):
        p[t, I["Hip"]] = (0, bend * 0.7, 0)
        p[t, I["Neck"]] = (0, -1 + bend * 0.7, 0)
        for side, sign in (("L", -1), ("R", 1)):
            p[t, I[f"{side}Shoulder"]] = (sign * 0.35, -0.9 + bend * 0.7, 0)
            p[t, I[f"{side}Hip"]] = (sign * 0.2, bend * 0.7, 0)
            p[t, I[f"{side}Knee"]] = (sign * (0.22 + bend * 0.55), 1.0, 0)
            p[t, I[f"{side}Ankle"]] = (sign * 0.25, 2.0, 0)
            p[t, I[f"{side}Heel"]] = (sign * 0.25, 2.05, 0)
            p[t, I[f"{side}BigToe"]] = (sign * 0.25, 1.9, 0)
    return p


def scorer(*, enable_ab_diagnostic=False):
    normal_2d = []
    normal_3d = []
    for amplitude in (0.90, 0.97, 1.03, 1.10):
        coords = sequence(amplitude)
        raw = coords[:, :, :2] * 100.0
        feat2d, summary = extract_2d_features(raw)
        normal_2d.append({"feat": feat2d, "summary": summary, "raw": raw})
        normal_3d.append({"feat": extract_all_features(coords)})
    return PostureScorer(normal_3d, normal_2d, enable_ab_diagnostic=enable_ab_diagnostic)


def realtime_matcher():
    refs = []
    for amplitude in (0.90, 0.97, 1.03, 1.10):
        raw = sequence(amplitude)[:, :, :2] * 100.0
        feat, summary = extract_2d_features(raw)
        refs.append({"raw": raw, "feat": feat, "summary": summary})
    return RealtimePostureMatcher(refs)


def visibility(frames):
    names = ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
             "LHeel", "RHeel", "LFootIndex", "RFootIndex")
    return [{name: 0.99 for name in names} for _ in range(frames)]


class PostureScoreTests(unittest.TestCase):
    def test_normal_like_is_high_and_all_scores_are_bounded(self):
        engine = scorer()
        coords = sequence(0.97)
        result = engine.score(coords[:, :, :2] * 100, coords, visibility(len(coords)),
                              baseline_knee=None, baseline_hip=None,
                              production_raw="정상", production_final="정상")
        self.assertTrue(result.score_valid)
        self.assertGreaterEqual(result.overall, 90.0)
        for value in result.components.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

    def test_far_trajectory_scores_lower_than_normal_like(self):
        engine = scorer()
        normal = sequence(0.97)
        far = sequence(2.0)
        normal_result = engine.score(normal[:, :, :2] * 100, normal, visibility(len(normal)),
                                     baseline_knee=None, baseline_hip=None)
        far_result = engine.score(far[:, :, :2] * 100, far, visibility(len(far)),
                                  baseline_knee=None, baseline_hip=None)
        self.assertLess(far_result.overall, normal_result.overall)

    def test_heel_score_is_independent_of_raw_heel_class(self):
        engine = scorer()
        coords = sequence(1.03)
        result = engine.score(coords[:, :, :2] * 100, coords, visibility(len(coords)),
                              baseline_knee=None, baseline_hip=None,
                              production_raw="발뒤꿈치오류", production_final="발뒤꿈치오류")
        self.assertGreaterEqual(result.components["heel_stability"], 90.0)

    def test_classification_unknown_can_still_have_valid_score(self):
        engine = scorer()
        coords = sequence(1.0)
        result = engine.score(coords[:, :, :2] * 100, coords, visibility(len(coords)),
                              baseline_knee=None, baseline_hip=None,
                              production_raw="엉덩이하방오류", production_final="자세추정불확실")
        self.assertTrue(result.score_valid)
        self.assertTrue(result.measurement_valid)

    def test_short_or_missing_measurement_is_invalid(self):
        engine = scorer()
        coords = sequence(1.0, frames=8)
        result = engine.score(coords[:, :, :2], coords, [], baseline_knee=None, baseline_hip=None)
        self.assertFalse(result.score_valid)
        self.assertFalse(result.measurement_valid)
        self.assertIsNone(result.overall)

    def test_realtime_match_is_bounded_and_phase_aware(self):
        engine = realtime_matcher()
        frame = sequence(1.0)[8, :, :2] * 100.0
        result = engine.update(frame, visibility(1)[0], phase="descend", baseline_ready=True)
        self.assertTrue(result["match_valid"])
        self.assertGreaterEqual(result["match_percent"], 0.0)
        self.assertLessEqual(result["match_percent"], 100.0)
        self.assertEqual(result["phase"], "descend")

    def test_realtime_match_requires_baseline_and_landmarks(self):
        engine = realtime_matcher()
        frame = sequence(1.0)[8, :, :2] * 100.0
        not_ready = engine.update(frame, visibility(1)[0], phase="descend", baseline_ready=False)
        invalid = engine.update(frame, {}, phase="descend", baseline_ready=True)
        self.assertFalse(not_ready["match_valid"])
        self.assertEqual(not_ready["invalid_reason"], "baseline_not_ready")
        self.assertFalse(invalid["match_valid"])

    def test_phase_change_resets_only_display_ema(self):
        engine = realtime_matcher()
        frame = sequence(1.0)[8, :, :2] * 100.0
        first = engine.update(frame, visibility(1)[0], phase="descend", baseline_ready=True)
        changed = engine.update(frame, visibility(1)[0], phase="bottom", baseline_ready=True)
        self.assertAlmostEqual(changed["match_percent"], changed["raw_match"])
        self.assertEqual(first["phase"], "descend")
        self.assertEqual(changed["phase"], "bottom")

    def test_match_aliases_keep_existing_storage_compatible(self):
        engine = scorer()
        coords = sequence(1.0)
        payload = engine.score(coords[:, :, :2] * 100, coords, visibility(len(coords)),
                               baseline_knee=None, baseline_hip=None).as_dict()
        self.assertEqual(payload["overall_match"], payload["overall"])
        self.assertEqual(payload["match_valid"], payload["score_valid"])

    def test_hybrid_score_has_version_method_and_ab_payload(self):
        engine = scorer(); coords = sequence(1.0)
        payload = engine.score(coords[:, :, :2] * 100, coords, visibility(len(coords)),
                               baseline_knee=None, baseline_hip=None).as_dict()
        self.assertEqual(payload["match_version"], 3)
        self.assertEqual(payload["match_method"], "hybrid_2d3d_robust_joint_scale_v2")
        self.assertNotIn("legacy", payload["diagnostic"])
        self.assertIn("hybrid", payload["diagnostic"])
        self.assertFalse(payload["diagnostic"]["ab_diagnostic_enabled"])

    def test_optional_legacy_ab_does_not_change_production_hybrid_score(self):
        coords = sequence(1.0)
        raw = coords[:, :, :2] * 100
        default = scorer().score(raw, coords, visibility(len(coords)),
                                 baseline_knee=None, baseline_hip=None)
        diagnostic = scorer(enable_ab_diagnostic=True).score(
            raw, coords, visibility(len(coords)), baseline_knee=None, baseline_hip=None
        )
        self.assertEqual(default.overall, diagnostic.overall)
        self.assertEqual(default.components, diagnostic.components)
        self.assertNotIn("legacy", default.diagnostic)
        self.assertIn("legacy", diagnostic.diagnostic)

    def test_depth_hip_and_knee_use_3d_not_2d(self):
        engine = scorer(); coords = sequence(1.0)
        raw = coords[:, :, :2] * 100
        first = engine.score(raw, coords, visibility(len(coords)), baseline_knee=None, baseline_hip=None)
        distorted = raw.copy()
        distorted[:, I["LKnee"], 0] += 500
        distorted[:, I["RKnee"], 0] -= 500
        second = engine.score(distorted, coords, visibility(len(coords)), baseline_knee=None, baseline_hip=None)
        for name in ("depth", "hip", "knee"):
            self.assertAlmostEqual(first.components[name], second.components[name])

    def test_component_source_contract(self):
        sources = scorer().reference_calibration["component_sources"]
        self.assertEqual(sources["depth"], "lifting_3d")
        self.assertEqual(sources["hip"], "lifting_3d")
        self.assertEqual(sources["knee"], "lifting_3d")
        self.assertEqual(sources["heel_stability"], "mediapipe_2d")
        self.assertEqual(sources["balance"], "mediapipe_2d")
        self.assertEqual(sources["trajectory"], "lifting_3d")

    def test_production_class_does_not_change_hybrid_match(self):
        engine = scorer(); coords = sequence(1.0); raw = coords[:, :, :2] * 100
        normal = engine.score(raw, coords, visibility(len(coords)), baseline_knee=None,
                              baseline_hip=None, production_raw="normal", production_final="normal")
        unknown = engine.score(raw, coords, visibility(len(coords)), baseline_knee=None,
                               baseline_hip=None, production_raw="hip-error", production_final="unknown")
        self.assertAlmostEqual(normal.overall, unknown.overall)
        self.assertEqual(normal.components, unknown.components)

    def test_large_heel_motion_lowers_heel_component(self):
        engine = scorer(); coords = sequence(1.0); raw = coords[:, :, :2] * 100
        normal = engine.score(raw, coords, visibility(len(coords)), baseline_knee=None, baseline_hip=None)
        moved = raw.copy()
        moved[len(moved)//3:2*len(moved)//3, I["LHeel"], 1] += 100
        moved[len(moved)//3:2*len(moved)//3, I["RHeel"], 1] += 100
        error = engine.score(moved, coords, visibility(len(coords)), baseline_knee=None, baseline_hip=None)
        self.assertLess(error.components["heel_stability"], normal.components["heel_stability"])

    def test_large_3d_deviation_lowers_depth_hip_or_knee(self):
        engine = scorer(); normal_coords = sequence(1.0); far = sequence(2.0)
        normal = engine.score(normal_coords[:, :, :2] * 100, normal_coords,
                              visibility(len(normal_coords)), baseline_knee=None, baseline_hip=None)
        error = engine.score(normal_coords[:, :, :2] * 100, far, visibility(len(far)),
                             baseline_knee=None, baseline_hip=None)
        self.assertLess(np.mean([error.components[x] for x in ("depth", "hip", "knee")]),
                        np.mean([normal.components[x] for x in ("depth", "hip", "knee")]))


if __name__ == "__main__":
    unittest.main()
