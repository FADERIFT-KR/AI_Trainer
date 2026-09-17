import unittest

import numpy as np
import torch

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.lifting_model import TemporalLiftingNet
from ai_trainer.lifting_parity import live_preprocess_2d, mean_leg_angles


class LiftingParityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.raw = rng.normal(size=(9, 18, 2)).astype(np.float32) * 20 + 200
        hip = COMMON_JOINT_NAMES.index("Hip")
        neck = COMMON_JOINT_NAMES.index("Neck")
        self.raw[:, hip] = [200, 300]
        self.raw[:, neck] = [200, 200]

    def test_same_raw_window_has_expected_live_shape(self):
        live, _ = live_preprocess_2d(self.raw)
        self.assertEqual(live.shape, (9, 18, 2))
        self.assertEqual(live.dtype, np.float32)

    def test_joint_order_is_the_model_common_order(self):
        self.assertEqual(COMMON_JOINT_NAMES[6:16], [
            "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
            "LHeel", "RHeel", "LBigToe", "RBigToe",
        ])

    def test_identical_scale_policy_has_numerical_input_parity(self):
        first, _ = live_preprocess_2d(self.raw, calib_frames=9)
        second, _ = live_preprocess_2d(self.raw, calib_frames=9)
        np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)

    def test_same_model_input_has_identical_3d_output(self):
        torch.manual_seed(3)
        model = TemporalLiftingNet(n_joints=18, hidden=128).eval()
        value, _ = live_preprocess_2d(self.raw)
        tensor = torch.from_numpy(value[None])
        with torch.no_grad():
            first = model(tensor).numpy()
            second = model(tensor).numpy()
        np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)

    def test_same_3d_output_has_identical_angles(self):
        coords = np.zeros((18, 3), dtype=np.float32)
        for index in range(18):
            coords[index] = [index % 3, index // 3, (index % 2) * 0.2]
        self.assertEqual(mean_leg_angles(coords), mean_leg_angles(coords.copy()))


if __name__ == "__main__":
    unittest.main()
