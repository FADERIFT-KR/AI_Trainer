"""Geometry and model-contract tests for the camera-ray lifting candidate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.camera_calibration import pixels_to_unit_rays, unmirror_pixels  # noqa: E402
from ai_trainer.ray_lifting import (  # noqa: E402
    ProjectiveCamera,
    ProjectionCalibrationError,
    RayTemporalLiftingNet,
    estimate_projective_camera,
)


class CameraRayGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix = np.array(
            [[900.0, 0.0, 640.0], [0.0, 880.0, 360.0], [0.0, 0.0, 1.0]]
        )

    def test_principal_point_maps_to_forward_unit_ray(self):
        ray = pixels_to_unit_rays(np.array([[640.0, 360.0]]), self.camera_matrix)
        np.testing.assert_allclose(ray, [[0.0, 0.0, 1.0]], atol=1e-9)

    def test_mirrored_pixel_recovers_the_same_physical_ray(self):
        original = np.array([[200.0, 300.0]])
        mirrored = np.array([[1279.0 - original[0, 0], original[0, 1]]])
        recovered = unmirror_pixels(mirrored, 1280)
        np.testing.assert_allclose(
            pixels_to_unit_rays(original, self.camera_matrix),
            pixels_to_unit_rays(recovered, self.camera_matrix),
            atol=1e-9,
        )

    def test_invalid_pixel_shape_is_rejected(self):
        with self.assertRaisesRegex(Exception, "shape"):
            pixels_to_unit_rays(np.zeros((3, 3)), self.camera_matrix)

    def test_dlt_recovers_a_synthetic_camera_and_reprojects(self):
        rng = np.random.default_rng(7)
        world = rng.uniform([-500.0, -700.0, -200.0], [500.0, 700.0, 400.0], size=(200, 3))
        rotation = np.eye(3)
        translation = np.array([15.0, -20.0, 2600.0])
        projection = self.camera_matrix @ np.column_stack((rotation, translation))
        known = ProjectiveCamera(self.camera_matrix, rotation, translation, projection)
        pixels = known.project(world)

        estimated = estimate_projective_camera(world, pixels, max_initial_points=200, max_refit_points=200)
        self.assertLess(float(estimated.reprojection_errors(world, pixels).max()), 1e-5)
        np.testing.assert_allclose(estimated.camera_matrix, self.camera_matrix, rtol=1e-5, atol=1e-5)

    def test_camera_relative_conversion_does_not_apply_translation(self):
        camera = ProjectiveCamera(
            self.camera_matrix,
            np.eye(3),
            np.array([10.0, 20.0, 30.0]),
            self.camera_matrix @ np.column_stack((np.eye(3), [10.0, 20.0, 30.0])),
        )
        np.testing.assert_allclose(
            camera.world_relative_to_camera(np.array([[1.0, 2.0, 3.0]])), [[1.0, 2.0, 3.0]]
        )


class RayLiftingModelTests(unittest.TestCase):
    def test_ray_temporal_model_accepts_three_dimensional_rays(self):
        model = RayTemporalLiftingNet(n_joints=18, hidden=16).eval()
        output = model(torch.zeros((2, 9, 18, 3)))
        self.assertEqual(tuple(output.shape), (2, 18, 3))

    def test_ray_temporal_model_rejects_pixel_feature_shape(self):
        model = RayTemporalLiftingNet(n_joints=18, hidden=16).eval()
        with self.assertRaisesRegex(ValueError, "Expected input shape"):
            model(torch.zeros((1, 9, 18, 2)))


if __name__ == "__main__":
    unittest.main()
