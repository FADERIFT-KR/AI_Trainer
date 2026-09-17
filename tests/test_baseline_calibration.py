import unittest

from ai_trainer.baseline_calibration import StandingBaselineCalibrator


def snapshot(hip=175.0, knee=176.0):
    return {
        "left_hip_angle_2d": hip - 1.0, "right_hip_angle_2d": hip + 1.0,
        "hip_angle_2d": hip, "left_knee_angle_2d": knee - 1.0,
        "right_knee_angle_2d": knee + 1.0, "knee_angle_2d": knee,
        "hip_angle_3d": hip - 2.0, "knee_angle_3d": knee - 2.0,
    }


class StandingBaselineCalibrationTests(unittest.TestCase):
    def calibrator(self):
        return StandingBaselineCalibrator(8, 163.47, 131.44, 154.09, 111.20, 5.64, 7.24)

    def test_a_eight_standing_frames_finalize(self):
        c = self.calibrator()
        for frame in range(8):
            row = c.observe(frame, snapshot(175 + frame * .1, 176 - frame * .1), 1.0, snapshot())
            self.assertTrue(row["accepted"])
        self.assertTrue(c.ready)
        self.assertGreater(c.result()["hip_2d"], 160)
        self.assertGreater(c.result()["knee_2d"], 130)

    def test_b_contradictory_hip_is_rejected(self):
        c = self.calibrator()
        row = c.observe(0, snapshot(10, 174), 1.0, snapshot())
        self.assertFalse(row["accepted"])
        self.assertIn("HIP_NOT_STANDING", row["reject_reason"])

    def test_c_moving_countdown_frames_are_rejected(self):
        c = self.calibrator()
        c.observe(0, snapshot(176, 176), 1.0, snapshot())
        row = c.observe(1, snapshot(168, 160), .9, snapshot())
        self.assertFalse(row["accepted"])
        self.assertIn("HIP_MOVING", row["reject_reason"])
        self.assertIn("KNEE_MOVING", row["reject_reason"])

    def test_d_invalid_frame_resets_consecutive_count(self):
        c = self.calibrator()
        for frame in range(4):
            c.observe(frame, snapshot(), 1.0, snapshot())
        c.observe(4, snapshot(10, 174), 1.0, snapshot())
        self.assertEqual(len(c.samples), 0)
        for frame in range(5, 13):
            c.observe(frame, snapshot(), 1.0, snapshot())
        self.assertTrue(c.ready)
        self.assertEqual(c.result()["frames"], list(range(5, 13)))

    def test_f_bottom_excursions_are_positive(self):
        c = self.calibrator()
        for frame in range(8):
            c.observe(frame, snapshot(), 1.0, snapshot())
        baseline = c.result()
        self.assertGreater(baseline["hip_2d"] - 85.0, 0)
        self.assertGreater(baseline["knee_2d"] - 55.0, 0)


if __name__ == "__main__":
    unittest.main()
