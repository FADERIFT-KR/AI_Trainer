import unittest

import numpy as np

from ai_trainer.game_ui.framing_check import check_active_tracking, check_framing


def pose(rows):
    points = np.zeros((33, 4), dtype=np.float32)
    points[:, :2] = (0.5, 0.5)
    points[:, 3] = 1.0
    for index, xy in rows.items():
        points[index, :2] = xy
    return points


def standing():
    return pose({
        0: (.5, .10), 11: (.42, .25), 12: (.58, .25),
        23: (.45, .50), 24: (.55, .50), 25: (.43, .68), 26: (.57, .68),
        27: (.42, .88), 28: (.58, .88), 29: (.41, .91), 30: (.59, .91),
    })


def squat():
    return pose({
        0: (.5, .30), 11: (.40, .40), 12: (.60, .40),
        23: (.44, .55), 24: (.56, .55), 25: (.34, .66), 26: (.66, .66),
        27: (.30, .73), 28: (.70, .73), 29: (.29, .76), 30: (.71, .76),
    })


class ActiveTrackingTests(unittest.TestCase):
    WIDTH, HEIGHT = 640, 480

    def baseline(self):
        return check_framing(standing(), self.WIDTH, self.HEIGHT).torso_scale

    def test_standing_too_far_is_not_ready(self):
        p = standing()
        p[:, :2] = .5 + (p[:, :2] - .5) * .4
        result = check_framing(p, self.WIDTH, self.HEIGHT)
        self.assertFalse(result.ok)
        self.assertIn("가까이", result.message)

    def test_standing_ready_passes_strict_prep_check(self):
        self.assertTrue(check_framing(standing(), self.WIDTH, self.HEIGHT).ok)

    def test_squat_bbox_shrink_is_tracked_in_active_mode(self):
        strict = check_framing(squat(), self.WIDTH, self.HEIGHT)
        active = check_active_tracking(squat(), self.WIDTH, self.HEIGHT, self.baseline())
        self.assertFalse(strict.ok)  # standing distance rule sees the natural shrink
        self.assertTrue(active.ok)
        self.assertLess(active.bbox_height, self.HEIGHT * .50)

    def test_deep_squat_with_visible_joints_remains_trackable(self):
        p = squat()
        p[0, 1], p[29, 1], p[30, 1] = .38, .72, .72
        self.assertTrue(check_active_tracking(p, self.WIDTH, self.HEIGHT, self.baseline()).ok)

    def test_actual_move_far_is_rejected_by_relative_torso_scale(self):
        p = standing()
        p[:, :2] = .5 + (p[:, :2] - .5) * .4
        result = check_active_tracking(p, self.WIDTH, self.HEIGHT, self.baseline())
        self.assertFalse(result.ok)
        self.assertIn("멀어", result.message)

    def test_screen_exit_or_tracking_loss_is_rejected(self):
        p = squat()
        p[27, 3] = 0.1
        self.assertFalse(check_active_tracking(p, self.WIDTH, self.HEIGHT, self.baseline()).ok)
        p = squat()
        p[:, 0] += .45
        self.assertFalse(check_active_tracking(p, self.WIDTH, self.HEIGHT, self.baseline()).ok)

    def test_standing_return_remains_trackable(self):
        self.assertTrue(check_active_tracking(standing(), self.WIDTH, self.HEIGHT, self.baseline()).ok)


if __name__ == "__main__":
    unittest.main()
