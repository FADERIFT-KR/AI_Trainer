import unittest

import numpy as np

from ai_trainer.bone_length_constraint import calibrate_leg_lengths, constrain_leg_lengths
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES


IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def standing_frame(scale=1.0):
    p = np.zeros((18, 3), dtype=float)
    for side, x in (("L", -.1), ("R", .1)):
        p[IDX[side+"Hip"]] = [x, 0, 0]
        p[IDX[side+"Knee"]] = [x, -.4*scale, 0]
        p[IDX[side+"Ankle"]] = [x, -.8*scale, 0]
        p[IDX[side+"Heel"]] = [x, -.82*scale, -.08*scale]
        p[IDX[side+"BigToe"]] = [x, -.82*scale, .15*scale]
    p[IDX["Hip"]] = [0, 0, 0]
    p[IDX["Neck"]] = [0, .5, 0]
    return p


class BoneLengthConstraintTests(unittest.TestCase):
    def test_lengths_are_fixed_and_directions_angles_preserved(self):
        calibration = calibrate_leg_lengths(np.stack([standing_frame(1+i*.001) for i in range(8)]))
        frame = standing_frame()
        frame[IDX["LKnee"]] = [-.25, -.22, -.31]
        frame[IDX["LAnkle"]] = [-.30, -.63, -.14]
        original = frame.copy()
        fixed = constrain_leg_lengths(frame, calibration)
        self.assertTrue(np.array_equal(frame, original))
        self.assertTrue(np.array_equal(fixed[IDX["Hip"]], original[IDX["Hip"]]))
        for side, expected in (("L", calibration.left), ("R", calibration.right)):
            ids = [IDX[side+n] for n in ("Hip","Knee","Ankle","Heel","BigToe")]
            h,k,a,heel,toe=ids
            got=[np.linalg.norm(fixed[k]-fixed[h]),np.linalg.norm(fixed[a]-fixed[k]),
                 np.linalg.norm(fixed[heel]-fixed[a]),np.linalg.norm(fixed[toe]-fixed[a]),
                 np.linalg.norm(fixed[toe]-fixed[heel])]
            np.testing.assert_allclose(got,[expected.thigh,expected.shin,expected.ankle_heel,
                                            expected.ankle_toe,expected.heel_toe],rtol=0,atol=1e-8)
            np.testing.assert_allclose((fixed[k]-fixed[h])/np.linalg.norm(fixed[k]-fixed[h]),
                                       (original[k]-original[h])/np.linalg.norm(original[k]-original[h]),atol=1e-9)
            np.testing.assert_allclose((fixed[a]-fixed[k])/np.linalg.norm(fixed[a]-fixed[k]),
                                       (original[a]-original[k])/np.linalg.norm(original[a]-original[k]),atol=1e-9)

    def test_rejects_nonstanding_calibration(self):
        pose=standing_frame();pose[IDX["LKnee"]]=[-.1,-.2,-.3]
        with self.assertRaisesRegex(ValueError,"standing frames"):
            calibrate_leg_lengths(np.stack([pose]*8))


if __name__ == "__main__":
    unittest.main()
