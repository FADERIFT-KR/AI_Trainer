import unittest

import numpy as np

from ai_trainer.game_ui.skeleton_panel import SkeletonViews
from tests.test_pose_timing import squat_pose


class SkeletonViewsTests(unittest.TestCase):
    def test_depth_is_visible_in_side_view_without_changing_front_view(self):
        sequence = np.stack([squat_pose(t) for t in np.linspace(0, 4, 60)])
        views = SkeletonViews(sequence, 480, 480)
        coords = sequence[30].copy()
        original = views.render(coords, estimated=True)
        coords[:, 2] += 0.2
        changed = views.render(coords, estimated=True)
        np.testing.assert_array_equal(original[:, :240], changed[:, :240])
        self.assertFalse(np.array_equal(original[:, 240:], changed[:, 240:]))
