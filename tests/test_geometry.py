from dataclasses import dataclass
import unittest

from src.geometry import joint_angle


@dataclass
class Point:
    x: float
    y: float


class JointAngleTests(unittest.TestCase):
    def test_right_angle(self) -> None:
        self.assertAlmostEqual(joint_angle(Point(1, 0), Point(0, 0), Point(0, 1)), 90.0)

    def test_straight_angle(self) -> None:
        self.assertAlmostEqual(joint_angle(Point(-1, 0), Point(0, 0), Point(1, 0)), 180.0)

    def test_overlapping_point_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            joint_angle(Point(0, 0), Point(0, 0), Point(1, 0))


if __name__ == "__main__":
    unittest.main()
