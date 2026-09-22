from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from ai_trainer.squat.camera_views import VIEW_LEFT, VIEW_RIGHT
from ai_trainer.core.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.squat.joint_feedback import compute_joint_scores, visible_joint_scores
from ai_trainer.squat.mt_stgcn import MultiTaskSTGCN, SquatErrorDiagnoser, adjacency_matrix
from ai_trainer.squat.online_dtw import OnlineSquatSession
from ai_trainer.squat.two_stage_squat import (
    NormalTemplateGate, build_normal_template, choose_threshold, dtw_path,
    mask_occluded_joints, normalize_track, visible_joint_weights, warp_to_reference,
)


def squat_track(length: int, pace: float = 1.0) -> np.ndarray:
    names = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
    x = np.zeros((length, 18, 3), dtype=np.float32)
    phase = np.linspace(0, 1, length) ** pace
    depth = 0.40 * np.sin(np.pi * phase) ** 2
    for side, sign in (("L", -1), ("R", 1)):
        hip_x = 0.12 * sign
        x[:, names[f"{side}Hip"]] = np.column_stack((np.full(length, hip_x), np.zeros(length), np.zeros(length)))
        x[:, names[f"{side}Knee"]] = np.column_stack((np.full(length, hip_x), -0.50 + depth * 0.4, depth * 0.3))
        x[:, names[f"{side}Ankle"]] = np.column_stack((np.full(length, hip_x), -1.0 + depth, np.zeros(length)))
        x[:, names[f"{side}Heel"]] = x[:, names[f"{side}Ankle"]]
        x[:, names[f"{side}BigToe"]] = x[:, names[f"{side}Ankle"]] + [0, 0, 0.15]
        x[:, names[f"{side}Shoulder"]] = np.array([hip_x, 0.55, 0])
    x[:, names["Neck"], 1] = 0.58
    return x


class TwoStageSquatTest(unittest.TestCase):
    def test_completed_repetition_uses_gate_then_graph_only_for_failures(self):
        class FakeDiagnoser:
            calls = 0

            def predict(self, aligned):
                self.calls += 1
                self_result = {"error": "발뒤꿈치오류", "body_part_probabilities": {"발목/발": 0.8}}
                return self_result

        def make_session(track, threshold, diagnoser):
            session = OnlineSquatSession(
                model=None, device=torch.device("cpu"),
                db_operational={"정상": [{}], "발뒤꿈치오류": [{}]},
                weights_cfg={"default_profile": "E_full_uniform"},
                normal_gate=NormalTemplateGate(squat_track(64), threshold),
                error_diagnoser=diagnoser,
            )
            session.aligned_seq = list(track)
            session.emit_offset = 0
            session.rep_start_idx = 0
            session.phase_boundaries_running = {}
            return session

        evidence = {"min_distance": 0.2, "best_detail": {"per_feature_contrib": {"test": 0.2}}}
        diagnoser = FakeDiagnoser()
        with patch("ai_trainer.squat.online_dtw.extract_all_features", return_value={}), \
             patch("ai_trainer.squat.online_dtw.resolve_weights", return_value={}), \
             patch("ai_trainer.squat.online_dtw.multi_reference_distance", return_value=evidence):
            good = make_session(squat_track(64), 0.05, diagnoser)
            good._finalize_rep(63)
            self.assertEqual(good.completed_reps[-1].predicted_class, "정상")
            self.assertEqual(good.completed_reps[-1].decision_stage, "dtw_pass")
            self.assertEqual(diagnoser.calls, 0)

            bad_track = squat_track(64)
            bad_track[:, COMMON_JOINT_NAMES.index("LAnkle"), 2] += 0.8
            bad = make_session(bad_track, 0.05, diagnoser)
            bad._finalize_rep(63)
            self.assertEqual(bad.completed_reps[-1].predicted_class, "발뒤꿈치오류")
            self.assertEqual(bad.completed_reps[-1].decision_stage, "mt_stgcn")
            self.assertEqual(diagnoser.calls, 1)

    def test_template_and_warp_are_invariant_to_scale_translation_and_pace(self):
        template = build_normal_template([squat_track(52, 0.8), squat_track(80, 1.2)], iterations=1)
        gate = NormalTemplateGate(template, threshold=0.08)
        user = squat_track(93, 1.1) * 2.4 + np.array([12, -3, 7], dtype=np.float32)
        result = gate.assess(user)
        self.assertTrue(result.passed)
        self.assertEqual(result.aligned.shape, (64, 18, 3))
        self.assertLess(result.distance, gate.threshold)
        self.assertAlmostEqual(float(np.median(np.linalg.norm(
            normalize_track(user)[:, 6] - normalize_track(user)[:, 8], axis=-1)
            + np.linalg.norm(normalize_track(user)[:, 8] - normalize_track(user)[:, 10], axis=-1))), 1.0, delta=0.05)

    def test_large_joint_deviation_fails_gate(self):
        reference = squat_track(64)
        changed = squat_track(70)
        changed[:, COMMON_JOINT_NAMES.index("LAnkle"), 2] += 0.8
        gate = NormalTemplateGate(reference, threshold=0.05)
        self.assertTrue(gate.assess(squat_track(70)).passed)
        self.assertFalse(gate.assess(changed).passed)

    def test_side_gate_ignores_camera_far_joints_and_uses_near_leg(self):
        reference = squat_track(64)
        changed = reference.copy()
        names = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
        for view, far, near in ((VIEW_LEFT, "R", "L"), (VIEW_RIGHT, "L", "R")):
            far_changed = changed.copy()
            for part in ("Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle", "Heel", "BigToe"):
                far_changed[:, names[f"{far}{part}"]] += [0.5, -0.7, 0.9]
            far_changed[:, names["Hip"]] += [0.3, 0.2, 0.1]
            far_changed[:, names["Neck"]] += [0.4, 0.1, 0.2]
            gate = NormalTemplateGate(reference, threshold=0.05, view=view)
            self.assertAlmostEqual(gate.assess(far_changed).distance,
                                   gate.assess(reference).distance, places=6)
            near_changed = reference.copy()
            near_changed[:, names[f"{near}Knee"], 2] += 0.8
            self.assertGreater(gate.assess(near_changed).distance, 0.05)
            self.assertEqual(visible_joint_weights(view)[names[f"{far}Knee"]], 0)
            self.assertGreater(visible_joint_weights(view)[names[f"{near}Knee"]], 0)

    def test_side_graph_input_masks_far_side_and_both_arms(self):
        track = np.ones((64, len(COMMON_JOINT_NAMES), 3), dtype=np.float32)
        names = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
        masked = mask_occluded_joints(track, VIEW_LEFT)
        self.assertEqual(float(masked[:, names["RKnee"]].sum()), 0.0)
        self.assertEqual(float(masked[:, names["LWrist"]].sum()), 0.0)
        self.assertEqual(float(masked[:, names["RWrist"]].sum()), 0.0)
        self.assertEqual(float(masked[:, names["LKnee"]].mean()), 1.0)
        np.testing.assert_array_equal(track, np.ones_like(track))

    def test_side_graph_prediction_is_invariant_to_hidden_joints(self):
        model = SquatErrorDiagnoser(MultiTaskSTGCN(width=8))
        base = squat_track(64)
        changed = base.copy()
        names = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
        for name in ("RShoulder", "RElbow", "RWrist", "RHip", "RKnee",
                     "RAnkle", "RHeel", "RBigToe", "LElbow", "LWrist", "Hip", "Neck"):
            changed[:, names[name]] += [0.3, -0.6, 0.8]
        original = model.predict(base, VIEW_LEFT)
        perturbed = model.predict(changed, VIEW_LEFT)
        self.assertEqual(original["error_probabilities"], perturbed["error_probabilities"])

    def test_side_joint_feedback_keeps_only_camera_near_angles(self):
        pose = squat_track(64)[10]
        scores = compute_joint_scores(pose, pose, phase="하강")
        self.assertEqual([score.name for score in visible_joint_scores(scores, VIEW_LEFT)],
                         ["Left Knee", "Left Hip", "Left Ankle"])
        self.assertEqual([score.name for score in visible_joint_scores(scores, VIEW_RIGHT)],
                         ["Right Knee", "Right Hip", "Right Ankle"])

    def test_side_without_new_artifacts_does_not_use_legacy_decision(self):
        session = OnlineSquatSession(
            model=None, device=torch.device("cpu"), db_operational={}, weights_cfg={},
            view_mode=VIEW_LEFT,
        )
        session.aligned_seq = list(squat_track(64))
        session.emit_offset = 0
        session.rep_start_idx = 0
        session._finalize_rep(63)
        self.assertEqual(session.completed_reps[-1].predicted_class, "판정 불확실")
        self.assertEqual(session.completed_reps[-1].decision_stage, "side_model_unavailable")

    def test_calibration_waits_for_consecutive_upright_frames(self):
        session = OnlineSquatSession(
            model=None, device=torch.device("cpu"), db_operational={}, weights_cfg={},
        )
        frame = squat_track(64)[0]
        waiting = session.push_frame_3d(frame, calibration_ready=False)
        self.assertEqual(waiting["status"], "waiting_for_calibration")
        self.assertIsNone(session.R_body)

        for expected in range(1, session.calib_frames):
            status = session.push_frame_3d(frame, calibration_ready=True)
            self.assertEqual(status["status"], "calibrating")
            self.assertEqual(status["calibration_samples"], expected)
        ready = session.push_frame_3d(frame, calibration_ready=True)
        self.assertEqual(ready["status"], "ok")
        self.assertIsNotNone(session.R_body)

    def test_dtw_path_covers_reference_and_threshold_uses_labels(self):
        a, b = squat_track(38), squat_track(64)
        distance, path = dtw_path(a, b)
        self.assertTrue(np.isfinite(distance))
        self.assertEqual(int(path[0, 1]), 0)
        self.assertEqual(int(path[-1, 1]), 63)
        self.assertEqual(warp_to_reference(a, path).shape, (64, 18, 3))
        threshold, stats = choose_threshold(np.array([0.1, 0.2, 0.7, 0.8]), np.array([1, 1, 0, 0]))
        self.assertAlmostEqual(threshold, 0.2)
        self.assertEqual(stats["error_false_pass_rate"], 0)

    def test_graph_has_bone_edges_and_two_trainable_heads(self):
        adjacency = adjacency_matrix()
        self.assertGreater(float(adjacency[6, 8]), 0)
        model = MultiTaskSTGCN(width=8)
        error, part = model(torch.randn(2, 3, 64, 18))
        self.assertEqual(tuple(error.shape), (2, 3))
        self.assertEqual(tuple(part.shape), (2, 4))
        (error.mean() + part.mean()).backward()
        self.assertIsNotNone(model.blocks[0].spatial.weight.grad)


if __name__ == "__main__":
    unittest.main()
