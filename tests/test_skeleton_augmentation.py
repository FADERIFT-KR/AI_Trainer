import unittest

import numpy as np

from ai_trainer.common_skeleton import PELVIS_IDX
from ai_trainer.skeleton_augmentation import (
    augment_skeleton_sequence,
    temporal_resample,
    yaw_rotate,
)


class SkeletonAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.coords = rng.normal(size=(20, 18, 3)).astype(np.float32)
        self.coords -= self.coords[:, PELVIS_IDX : PELVIS_IDX + 1]

    def test_temporal_resample_changes_duration_and_preserves_endpoints(self) -> None:
        result = temporal_resample(self.coords, 1.25)
        self.assertEqual(result.shape, (25, 18, 3))
        np.testing.assert_allclose(result[0], self.coords[0])
        np.testing.assert_allclose(result[-1], self.coords[-1])

    def test_yaw_rotation_preserves_joint_distances(self) -> None:
        result = yaw_rotate(self.coords, 8.0)
        before = np.linalg.norm(self.coords[:, 0] - self.coords[:, 1], axis=-1)
        after = np.linalg.norm(result[:, 0] - result[:, 1], axis=-1)
        np.testing.assert_allclose(after, before, atol=1e-6)

    def test_augmentation_is_deterministic_and_keeps_pelvis_at_origin(self) -> None:
        first = augment_skeleton_sequence(
            self.coords,
            duration_scale=0.8,
            yaw_deg=5.0,
            jitter_std=0.005,
            seed=11,
        )
        second = augment_skeleton_sequence(
            self.coords,
            duration_scale=0.8,
            yaw_deg=5.0,
            jitter_std=0.005,
            seed=11,
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first[:, PELVIS_IDX], 0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
