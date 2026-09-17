import unittest

import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.online_dtw import OnlineSquatSession, validate_production_candidate
from tests.test_posture_score import scorer


I = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}


def symmetric_pose(knee_x=0.0):
    pose = np.zeros((18, 3), dtype=float)
    pose[I["Neck"]] = (0.0, -1.0, 0.0)
    for side, sign in (("L", -1.0), ("R", 1.0)):
        pose[I[f"{side}Hip"]] = (sign * 0.2, 0.0, 0.0)
        pose[I[f"{side}Knee"]] = (sign * (0.2 + knee_x), 1.0, 0.0)
        pose[I[f"{side}Ankle"]] = (sign * 0.2, 2.0, 0.0)
    return pose


class AccuracyPatch2ATests(unittest.TestCase):
    def test_2d_and_3d_use_same_included_angle_definition(self):
        pose = symmetric_pose(0.4)
        values = OnlineSquatSession._pose_angle_snapshot(pose[:, :2], pose)
        self.assertAlmostEqual(values["hip_angle_2d"], values["hip_angle_3d"], places=5)
        self.assertAlmostEqual(values["knee_angle_2d"], values["knee_angle_3d"], places=5)

    def test_standing_angles_are_greater_than_bent_angles(self):
        standing = symmetric_pose(0.0)
        bent = symmetric_pose(0.8)
        a = OnlineSquatSession._pose_angle_snapshot(standing[:, :2], standing)
        b = OnlineSquatSession._pose_angle_snapshot(bent[:, :2], bent)
        self.assertGreater(a["hip_angle_3d"], b["hip_angle_3d"])
        self.assertGreater(a["knee_angle_3d"], b["knee_angle_3d"])

    def test_left_and_right_are_symmetric(self):
        values = OnlineSquatSession._pose_angle_snapshot(
            symmetric_pose(0.5)[:, :2], symmetric_pose(0.5)
        )
        self.assertAlmostEqual(values["left_hip_angle_3d"], values["right_hip_angle_3d"])
        self.assertAlmostEqual(values["left_knee_angle_3d"], values["right_knee_angle_3d"])

    def test_joint_score_scale_uses_normal_reference_p90(self):
        calibration = scorer().reference_calibration["components"]
        for name in ("hip", "knee"):
            self.assertEqual(
                calibration[name]["score_scale"],
                calibration[name]["intra_nearest_distance"]["p90"],
            )

    def test_consistency_mismatch_no_longer_vetoes_otherwise_valid_normal(self):
        self.assertEqual(
            validate_production_candidate("정상", depth_2d_pass=True, consistency_pass=False),
            "정상",
        )

    def test_existing_depth_guard_is_still_active(self):
        self.assertEqual(
            validate_production_candidate("엉덩이하방오류", depth_2d_pass=True,
                                          consistency_pass=False),
            "자세추정불확실",
        )


if __name__ == "__main__":
    unittest.main()
