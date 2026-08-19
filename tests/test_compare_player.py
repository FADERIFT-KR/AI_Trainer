import unittest

from src.compare_player import compare_measurements, frame_for_elapsed, reference_measurement
from src.pose_tracker import PoseMeasurement


class ComparePlayerTests(unittest.TestCase):
    def test_reference_measurement_for_straight_legs(self) -> None:
        points = {
            "LShoulder": (0, -1), "LHip": (0, 0), "LKnee": (0, 1), "LAnkle": (0, 2),
            "RShoulder": (1, -1), "RHip": (1, 0), "RKnee": (1, 1), "RAnkle": (1, 2),
        }
        value = reference_measurement(points)
        self.assertAlmostEqual(value.left_knee_angle, 180.0)
        self.assertAlmostEqual(value.right_knee_angle, 180.0)
        self.assertAlmostEqual(value.left_hip_angle, 180.0)
        self.assertAlmostEqual(value.right_hip_angle, 180.0)

    def test_identical_measurements_are_full_match(self) -> None:
        value = PoseMeasurement(90.0, 91.0, 80.0, 82.0)
        comparison = compare_measurements(value, value)
        self.assertEqual(comparison.similarity, 100.0)
        self.assertEqual(comparison.mean_error, 0.0)

    def test_comparison_reports_largest_difference(self) -> None:
        reference = PoseMeasurement(100.0, 100.0, 100.0, 100.0)
        live = PoseMeasurement(100.0, 70.0, 100.0, 100.0)
        comparison = compare_measurements(reference, live)
        self.assertEqual(comparison.largest_error_name, "right knee")
        self.assertEqual(comparison.largest_error, 30.0)
        self.assertAlmostEqual(comparison.similarity, 91.6666667)

    def test_frame_playback_uses_elapsed_time_and_loops(self) -> None:
        self.assertEqual(frame_for_elapsed(1.0, fps=30.0, frame_count=214), 30)
        self.assertEqual(frame_for_elapsed(7.2, fps=30.0, frame_count=214), 2)

    def test_frame_playback_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            frame_for_elapsed(1.0, fps=0.0, frame_count=214)


if __name__ == "__main__":
    unittest.main()
