import unittest
import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.game_ui.pose_bridge import CommonSkeleton3DBridge


class FootTrackingTests(unittest.TestCase):
    def test_occluded_foot_follows_ankle_during_squat(self):
        points = np.zeros((33, 4))
        points[:, 3] = 1
        points[27, :3] = [-0.1, 0.9, 0]
        points[29, :3] = [-0.1, 0.94, -0.05]
        points[31, :3] = [-0.1, 0.94, 0.15]
        bridge = CommonSkeleton3DBridge()
        first, _, _ = bridge.update(points)
        points[[27, 29, 31], 1] -= 0.4
        points[[29, 31], 3] = 0.1
        second, frozen, _ = bridge.update(points)
        ankle = COMMON_JOINT_NAMES.index("LAnkle")
        for name in ("LHeel", "LBigToe"):
            foot = COMMON_JOINT_NAMES.index(name)
            self.assertTrue(frozen[foot])
            np.testing.assert_allclose(second[foot] - second[ankle], first[foot] - first[ankle])
        self.assertLess(second[ankle, 1], first[ankle, 1] - 0.2)

    def test_visible_heel_lift_is_not_flattened(self):
        points = np.zeros((33, 4))
        points[:, 3] = 1
        points[29, :3] = [0, 0.04, -0.05]
        bridge = CommonSkeleton3DBridge()
        first, _, _ = bridge.update(points)
        points[29, 1] -= 0.1
        for _ in range(10):
            second, frozen, _ = bridge.update(points)
        heel = COMMON_JOINT_NAMES.index("LHeel")
        self.assertFalse(frozen[heel])
        self.assertLess(second[heel, 1], first[heel, 1] - 0.08)
