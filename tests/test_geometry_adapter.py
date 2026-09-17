import unittest
import numpy as np

from ai_trainer.geometry_adapter import adapt_mediapipe_to_training_domain
from ai_trainer.lifting_parity import mean_leg_angles


class GeometryAdapterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.sequence = rng.normal(size=(12, 18, 2)).astype(np.float32)

    def test_shape_finite_and_deterministic(self):
        a = adapt_mediapipe_to_training_domain(self.sequence)
        b = adapt_mediapipe_to_training_domain(self.sequence)
        self.assertEqual(a.shape, (12, 18, 2))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a, b)

    def test_disabled_is_exact_identity(self):
        np.testing.assert_array_equal(
            adapt_mediapipe_to_training_domain(self.sequence, enabled=False), self.sequence
        )

    def test_hip_root_and_temporal_order_are_preserved(self):
        adapted = adapt_mediapipe_to_training_domain(self.sequence)
        np.testing.assert_allclose(adapted[:, 16], self.sequence[:, 16])
        self.assertEqual(len(adapted), len(self.sequence))

    def test_leg_angles_are_not_forced(self):
        adapted = adapt_mediapipe_to_training_domain(self.sequence)
        before = np.array([mean_leg_angles(frame) for frame in self.sequence])
        after = np.array([mean_leg_angles(frame) for frame in adapted])
        self.assertGreater(float(np.std(after[:, 1])), 0.0)
        self.assertFalse(np.allclose(after[:, 1], 75.0))
        self.assertGreater(float(np.ptp(after[:, 1])), 0.1)


if __name__ == "__main__":
    unittest.main()
