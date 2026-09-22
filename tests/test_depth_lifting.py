from __future__ import annotations

import unittest

import torch

from ai_trainer.squat.depth_lifting import TEMPORAL_WINDOW, VideoPose3DDepthNet, depth_aware_lifting_loss


class DepthLiftingTest(unittest.TestCase):
    def test_videopose3d_variant_outputs_only_center_pose(self):
        model = VideoPose3DDepthNet(n_joints=18, channels=16, dropout=0.0).eval()
        output = model(torch.zeros((2, TEMPORAL_WINDOW, 18, 2)))
        self.assertEqual(tuple(output.shape), (2, 18, 3))

    def test_rejects_wrong_temporal_window(self):
        model = VideoPose3DDepthNet(n_joints=18, channels=16)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 9, 18, 2)))

    def test_depth_loss_weights_z_and_keeps_bone_term(self):
        target = torch.zeros((1, 18, 3))
        prediction = target.clone()
        prediction[..., 2] = 1.0
        loss, components = depth_aware_lifting_loss(prediction, target)
        self.assertGreater(float(loss), 1.9)
        self.assertAlmostEqual(float(components["z_mse"]), 1.0)


if __name__ == "__main__":
    unittest.main()
