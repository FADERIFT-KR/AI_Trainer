import unittest

import numpy as np

from ai_trainer.game_ui.squat_form import assess_squat_form


def _landmarks(*, squat: bool = False, bad_feet: bool = False) -> np.ndarray:
    p = np.zeros((33, 4), dtype=np.float32)
    p[:, 3] = 1.0
    p[0, :2] = (0.50, 0.10)
    p[11, :2], p[12, :2] = (0.40, 0.20), (0.60, 0.20)
    p[23, :2], p[24, :2] = (0.44, 0.50 if not squat else 0.68), (0.56, 0.50 if not squat else 0.68)
    p[25, :2], p[26, :2] = (0.43, 0.68), (0.57, 0.68)
    p[27, :2], p[28, :2] = (0.40, 0.80), (0.60, 0.80)
    p[29, :2], p[30, :2] = (0.38, 0.82), (0.62, 0.82)
    # ankle->toe vectors are about 20 degrees outward from the downward shin axis.
    p[31, :2], p[32, :2] = (0.350, 0.866), (0.650, 0.866)
    if bad_feet:
        p[31, :2], p[32, :2] = (0.40, 0.85), (0.60, 0.85)
    return p


class SquatFormTests(unittest.TestCase):
    def test_preparation_accepts_natural_stance(self):
        self.assertEqual(assess_squat_form(_landmarks(), 640, 480, "prep").warnings, ())

    def test_preparation_reports_toe_or_contact_problem(self):
        warnings = assess_squat_form(_landmarks(bad_feet=True), 640, 480, "prep").warnings
        self.assertTrue(any("각도" in warning for warning in warnings))

    def test_bottom_reports_insufficient_depth(self):
        warnings = assess_squat_form(_landmarks(), 640, 480, "bottom").warnings
        self.assertTrue(any("무릎 높이" in warning for warning in warnings))

    def test_visibility_warning_is_not_distance_warning(self):
        p = _landmarks()
        p[29, 3] = 0.1
        warnings = assess_squat_form(p, 640, 480, "descend").warnings
        self.assertTrue(any("발이 잘 보이지" in warning for warning in warnings))

    def test_world_angle_uses_camera_direction_as_zero(self):
        p = _landmarks()
        world = np.zeros((33, 4), dtype=np.float32)
        world[:, 3] = 1.0
        # Both toes point straight toward the camera (depth axis), so angle is 0°.
        world[27, :3], world[31, :3] = (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)
        world[28, :3], world[32, :3] = (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)
        result = assess_squat_form(p, 640, 480, "prep", world)
        self.assertEqual(result.foot_angles, (0.0, 0.0))

    def test_world_angle_sign_is_left_ccw_and_right_cw(self):
        p = _landmarks()
        world = np.zeros((33, 4), dtype=np.float32)
        world[:, 3] = 1.0
        world[27, :3], world[31, :3] = (0.0, 0.0, 0.0), (-0.3, 0.0, -1.0)
        world[28, :3], world[32, :3] = (0.0, 0.0, 0.0), (0.3, 0.0, -1.0)
        left, right = assess_squat_form(p, 640, 480, "prep", world).foot_angles
        self.assertLess(left, 0.0)
        self.assertGreater(right, 0.0)

    def test_object_side_labels_control_sign_not_camera_screen_side(self):
        p = _landmarks()
        world = np.zeros((33, 4), dtype=np.float32)
        world[:, 3] = 1.0
        # Deliberately swap x directions; anatomical L/R still define the sign.
        world[27, :3], world[31, :3] = (0.0, 0.0, 0.0), (0.3, 0.0, -1.0)
        world[28, :3], world[32, :3] = (0.0, 0.0, 0.0), (-0.3, 0.0, -1.0)
        left, right = assess_squat_form(p, 640, 480, "prep", world).foot_angles
        self.assertLess(left, 0.0)
        self.assertGreater(right, 0.0)


if __name__ == "__main__":
    unittest.main()
