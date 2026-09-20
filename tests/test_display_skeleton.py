"""Side-view display correction must not alter the analysis coordinates."""
from __future__ import annotations

import unittest

import numpy as np

from ai_trainer.camera_views import VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.game_ui.display_skeleton import ImageGuidedSkeletonDisplay


IDX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}


class ImageGuidedDisplayTests(unittest.TestCase):
    def test_side_leg_follows_visible_image_order_at_squat_bottom(self):
        image = np.full((len(COMMON_JOINT_NAMES), 2), (100.0, 100.0))
        image[IDX["Hip"]] = (100.0, 100.0)
        image[IDX["RHip"]] = (100.0, 100.0)
        image[IDX["RKnee"]] = (160.0, 95.0)
        image[IDX["RAnkle"]] = (140.0, 200.0)
        pose = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=float)
        pose[IDX["RKnee"]] = (0.1, -0.5, -0.1)
        pose[IDX["RAnkle"]] = (0.1, -0.1, -0.3)  # erroneous 3D ankle above knee
        original = pose.copy()

        corrected = ImageGuidedSkeletonDisplay(VIEW_RIGHT).update(image, pose)
        self.assertLess(corrected[IDX["RAnkle"], 1], corrected[IDX["RKnee"], 1])
        self.assertGreater(-corrected[IDX["RKnee"], 2], -corrected[IDX["RAnkle"], 2])
        np.testing.assert_array_equal(pose, original)

    def test_front_view_is_not_changed(self):
        image = np.zeros((len(COMMON_JOINT_NAMES), 2), dtype=float)
        pose = np.ones((len(COMMON_JOINT_NAMES), 3), dtype=float)
        self.assertIs(ImageGuidedSkeletonDisplay(VIEW_FRONT).update(image, pose), pose)

    def test_occluded_far_wrist_uses_visible_arm_without_altering_input(self):
        image = np.full((len(COMMON_JOINT_NAMES), 2), (100.0, 100.0))
        image[IDX["LHip"]] = (100.0, 100.0)
        image[IDX["LKnee"]] = (100.0, 180.0)
        image[IDX["LAnkle"]] = (100.0, 260.0)
        pose = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=float)
        pose[IDX["LShoulder"]] = (0.2, 0.6, 0.0)
        pose[IDX["LElbow"]] = (0.35, 0.35, 0.1)
        pose[IDX["LWrist"]] = (0.4, 0.1, 0.2)
        pose[IDX["RShoulder"]] = (-0.2, 0.6, 0.0)
        pose[IDX["RElbow"]] = (-3.0, 2.0, -1.0)
        pose[IDX["RWrist"]] = (-5.0, 3.0, -2.0)
        original = pose.copy()
        world = np.zeros((33, 4), dtype=float)
        world[:, 3] = 1.0
        world[[14, 16], 3] = 0.05
        corrected = ImageGuidedSkeletonDisplay(VIEW_LEFT).update(image, pose, world)
        np.testing.assert_allclose(corrected[IDX["RElbow"]], (-0.35, 0.35, 0.1))
        np.testing.assert_allclose(corrected[IDX["RWrist"]], (-0.4, 0.1, 0.2))
        np.testing.assert_array_equal(pose, original)

    def test_visible_far_arm_is_preserved_in_right_view(self):
        image = np.full((len(COMMON_JOINT_NAMES), 2), (100.0, 100.0))
        image[IDX["RHip"]] = (100.0, 100.0)
        image[IDX["RKnee"]] = (100.0, 180.0)
        image[IDX["RAnkle"]] = (100.0, 260.0)
        pose = np.arange(len(COMMON_JOINT_NAMES) * 3, dtype=float).reshape(-1, 3)
        world = np.zeros((33, 4), dtype=float)
        world[:, 3] = 1.0
        corrected = ImageGuidedSkeletonDisplay(VIEW_RIGHT).update(image, pose, world)
        np.testing.assert_array_equal(corrected[IDX["LElbow"]], pose[IDX["LElbow"]])
        np.testing.assert_array_equal(corrected[IDX["LWrist"]], pose[IDX["LWrist"]])

    def test_visible_wrist_depth_spike_is_limited_when_image_is_still(self):
        image = np.full((len(COMMON_JOINT_NAMES), 2), (100.0, 100.0))
        image[IDX["LHip"]] = (100.0, 100.0)
        image[IDX["LKnee"]] = (100.0, 180.0)
        image[IDX["LAnkle"]] = (100.0, 260.0)
        world = np.zeros((33, 4), dtype=float)
        world[:, 3] = 1.0
        world[[14, 16], 3] = 0.1
        pose = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=float)
        pose[IDX["LWrist"], 0] = -0.3
        corrector = ImageGuidedSkeletonDisplay(VIEW_LEFT)
        first = corrector.update(image, pose, world)
        pose[IDX["LWrist"], 0] = 0.3
        second = corrector.update(image, pose, world)
        self.assertAlmostEqual(second[IDX["LWrist"], 0] - first[IDX["LWrist"], 0], 0.12)
        self.assertAlmostEqual(second[IDX["RWrist"], 0] - first[IDX["RWrist"], 0], -0.12)


if __name__ == "__main__":
    unittest.main()
