import unittest

import numpy as np
import torch

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.game_ui.one_euro_filter import OneEuroFilter
from ai_trainer.online_dtw import OnlineSquatSession, RepResult
from ai_trainer.pose_sampling import PoseSampler


def squat_pose(t):
    depth = np.clip(min(t - 0.8, 3.2 - t), 0, 1)
    angle = depth * np.pi / 3
    pose = np.zeros((18, 3))
    ix = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
    for side, x in (("L", -0.1), ("R", 0.1)):
        pose[ix[side + "Hip"]] = [x, 0, 0]
        pose[ix[side + "Knee"]] = [x, -0.5 * np.cos(angle), 0.5 * np.sin(angle)]
        pose[ix[side + "Ankle"]] = [x, -np.cos(angle), 0]
        pose[ix[side + "Shoulder"]] = [2*x, 0.5, 0]
    pose[ix["Neck"]] = [0, 0.5, 0]
    return pose


class TimingTests(unittest.TestCase):
    def test_sampling_same_linear_motion_at_10_and_30_fps(self):
        outputs = []
        for fps in (10, 30):
            sampler = PoseSampler()
            result = []
            for t in np.linspace(0, 1, fps + 1):
                result.extend(sampler.push(np.full((18, 3), t), float(t)))
            outputs.append(np.stack(result))
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-12)
        self.assertEqual(len(outputs[0]), 31)

    def test_gap_is_not_filled_and_timestamp_order_is_validated(self):
        sampler = PoseSampler()
        sampler.push(squat_pose(0), 0)
        self.assertEqual(len(sampler.push(squat_pose(2), 2)), 1)
        self.assertTrue(sampler.discontinuity)
        with self.assertRaises(ValueError):
            sampler.push(squat_pose(2), 2)

    def test_first_visible_sample_does_not_blend_with_unobserved_value(self):
        smoother = OneEuroFilter(1, 3)
        smoother(np.full((1, 3), np.nan), np.array([False]), timestamp=0)
        result = smoother(np.ones((1, 3)), np.array([True]), timestamp=0.1)
        np.testing.assert_array_equal(result, np.ones((1, 3)))

    def test_filter_uses_actual_elapsed_time(self):
        smoother = OneEuroFilter(1, 1, beta=0)
        smoother(np.zeros((1, 1)), np.array([True]), timestamp=0)
        actual = smoother(np.ones((1, 1)), np.array([True]), timestamp=0.1)
        r = 2 * np.pi * 0.8 * 0.1
        self.assertAlmostEqual(float(actual[0, 0]), r / (1+r))

    def test_rep_timing_is_stable_across_capture_rates(self):
        completions = []
        for fps in (10, 30):
            session = OnlineSquatSession(torch.nn.Identity(), torch.device("cpu"), {}, {})
            session._partial_online_distance = lambda t: None
            session._joint_feedback_frame = lambda t: None
            def complete(end_t):
                session.completed_reps.append(RepResult(0, (session.rep_start_idx, end_t), "test", {}, None, []))
            session._finalize_rep = complete
            observed_results = []
            for t in np.arange(0, 4.1, 1 / fps):
                status = session.push_frame_3d(squat_pose(float(t)), timestamp=float(t))
                if status and status.get("completed_rep"):
                    observed_results.append(status["completed_rep"])
            self.assertEqual(len(session.completed_reps), 1)
            self.assertEqual(len(observed_results), 1)  # result survives intermediate samples
            completions.append(session.completed_reps[0].frame_range)
        np.testing.assert_allclose(completions[0], completions[1], atol=2)

    def test_crouching_does_not_calibrate_and_outage_discards_partial_rep(self):
        session = OnlineSquatSession(torch.nn.Identity(), torch.device("cpu"), {}, {})
        session._partial_online_distance = lambda t: None
        session._joint_feedback_frame = lambda t: None
        for t in np.arange(0, 0.5, 0.1):
            status = session.push_frame_3d(squat_pose(2), timestamp=float(t))
        self.assertIsNone(session.R_body)
        self.assertEqual(status["status"], "calibrating")
        for t in np.arange(0.5, 1.1, 0.1):
            session.push_frame_3d(squat_pose(0), timestamp=float(t))
        self.assertIsNotNone(session.R_body)
        session.state = "descend"
        status = session.push_frame_3d(squat_pose(2), timestamp=2)
        self.assertEqual(status["status"], "calibrating")
        self.assertEqual(session.state, "prep")
        self.assertIsNone(session.R_body)
