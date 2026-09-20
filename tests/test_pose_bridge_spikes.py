"""Regression tests for live 3D knee tracking at the squat reversal."""
from __future__ import annotations

import unittest

import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.game_ui.pose_bridge import CommonSkeleton3DBridge


INDEX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def landmarks(depth: float = 0.0, left_knee_offset: float = 0.0,
              left_knee_visibility: float = 1.0) -> np.ndarray:
    points = np.zeros((33, 4), dtype=np.float64)
    points[:, 3] = 1.0
    points[11, :3] = (-0.18, 0.65, 0.0)
    points[12, :3] = (0.18, 0.65, 0.0)
    points[23, :3] = (-0.13, 0.0, 0.0)
    points[24, :3] = (0.13, 0.0, 0.0)
    points[25, :3] = (-0.13 + left_knee_offset, -0.48 + 0.1 * depth, 0.05 + left_knee_offset)
    points[26, :3] = (0.13, -0.48 + 0.1 * depth, 0.05)
    points[27, :3] = (-0.13, -0.98 + depth, 0.0)
    points[28, :3] = (0.13, -0.98 + depth, 0.0)
    points[29, :3] = points[27, :3]
    points[30, :3] = points[28, :3]
    points[31, :3] = points[27, :3] + (0, 0, 0.15)
    points[32, :3] = points[28, :3] + (0, 0, 0.15)
    points[25, 3] = left_knee_visibility
    return points


class Live3DSpikeTests(unittest.TestCase):
    def test_single_reacquisition_spike_does_not_jump_at_bottom(self):
        bridge = CommonSkeleton3DBridge(min_visibility=0.5)
        outputs = []
        for depth in np.r_[np.linspace(0, 0.25, 12), [0.25, 0.25]]:
            outputs.append(bridge.update(landmarks(float(depth)))[0])
        for _ in range(2):
            outputs.append(bridge.update(landmarks(0.25, left_knee_visibility=0.1))[0])
        bad, frozen, _ = bridge.update(landmarks(0.25, left_knee_offset=0.35))
        outputs.append(bad)
        outputs.append(bridge.update(landmarks(0.25))[0])
        left = np.stack(outputs)[:, INDEX["LKnee"]]
        right = np.stack(outputs)[:, INDEX["RKnee"]]
        self.assertTrue(frozen[INDEX["LKnee"]])
        self.assertLess(float(np.linalg.norm(left[-2] - left[-3])), 0.07)
        self.assertLess(float(np.linalg.norm(left[-1] - left[-2])), 0.07)
        self.assertLess(float(np.max(np.linalg.norm(np.diff(right, axis=0), axis=1))), 0.07)

    def test_genuine_squat_descent_is_not_frozen(self):
        bridge = CommonSkeleton3DBridge(min_visibility=0.5)
        outputs = [bridge.update(landmarks(float(d)))[0] for d in np.linspace(0, 0.32, 20)]
        left = np.stack(outputs)[:, INDEX["LKnee"]]
        self.assertGreater(float(np.linalg.norm(left[-1] - left[0])), 0.02)

    def test_left_and_right_hip_use_symmetric_spike_guard(self):
        changes = []
        for landmark_index, joint_name, sign in ((23, "LHip", -1), (24, "RHip", 1)):
            bridge = CommonSkeleton3DBridge()
            for _ in range(10):
                before = bridge.update(landmarks())[0]
            bad = landmarks()
            bad[landmark_index, 0] += sign * 0.22
            after, frozen, _ = bridge.update(bad)
            changes.append(float(np.linalg.norm(after[INDEX[joint_name]] - before[INDEX[joint_name]])))
            self.assertTrue(frozen[INDEX[joint_name]])
            np.testing.assert_allclose((after[INDEX["LHip"]] + after[INDEX["RHip"]]) / 2,
                                       after[INDEX["Hip"]], atol=1e-10)
        self.assertLess(max(changes), 0.07)
        self.assertAlmostEqual(changes[0], changes[1], delta=0.01)

    def test_persistent_plausible_knee_change_is_followed(self):
        bridge = CommonSkeleton3DBridge()
        outputs = [bridge.update(landmarks())[0] for _ in range(10)]
        outputs.extend(bridge.update(landmarks(left_knee_offset=0.12))[0] for _ in range(10))
        left = np.stack(outputs)[:, INDEX["LKnee"]]
        self.assertGreater(float(np.linalg.norm(left[-1] - left[9])), 0.08)
        self.assertLess(float(np.max(np.linalg.norm(np.diff(left, axis=0), axis=1))), 0.08)

    def test_occluded_ankle_depth_jump_keeps_foot_attached(self):
        bridge = CommonSkeleton3DBridge(stabilize_feet=True)
        for _ in range(10):
            before = bridge.update(landmarks())[0]
        bad = landmarks()
        for index in (27, 29, 31):
            bad[index, 2] -= 0.30
        after, frozen, _ = bridge.update(bad)
        ankle = INDEX["LAnkle"]
        heel = INDEX["LHeel"]
        toe = INDEX["LBigToe"]
        self.assertTrue(frozen[ankle])
        self.assertLess(float(np.linalg.norm(after[ankle] - before[ankle])), 0.08)
        self.assertLess(abs(float(np.linalg.norm(after[heel] - after[ankle])
                                  - np.linalg.norm(before[heel] - before[ankle]))), 0.05)
        self.assertLess(abs(float(np.linalg.norm(after[toe] - after[ankle])
                                  - np.linalg.norm(before[toe] - before[ankle]))), 0.05)
        knee = INDEX["LKnee"]
        self.assertAlmostEqual(float(np.linalg.norm(after[knee] - after[ankle])),
                               float(np.linalg.norm(before[knee] - before[ankle])), delta=0.01)
        outputs = [after]
        outputs.extend(bridge.update(bad)[0] for _ in range(8))
        ankle_steps = np.linalg.norm(np.diff(np.stack(outputs)[:, ankle], axis=0), axis=1)
        self.assertLess(float(ankle_steps.max()), 0.13)

    def test_display_stabilizer_is_separate_from_analysis_bridge(self):
        analysis = CommonSkeleton3DBridge()
        display = CommonSkeleton3DBridge(stabilize_feet=True)
        for _ in range(10):
            analysis.update(landmarks())
            display.update(landmarks())
        bad = landmarks()
        for index in (27, 29, 31):
            bad[index, 2] -= 0.30
        analytic, analytic_frozen, _ = analysis.update(bad)
        visual, visual_frozen, _ = display.update(bad)
        ankle = INDEX["LAnkle"]
        self.assertFalse(analytic_frozen[ankle])
        self.assertTrue(visual_frozen[ankle])
        self.assertGreater(float(np.linalg.norm(analytic[ankle] - visual[ankle])), 0.10)


if __name__ == "__main__":
    unittest.main()
