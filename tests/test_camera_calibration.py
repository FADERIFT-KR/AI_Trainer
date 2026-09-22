"""Headless tests for ChArUco camera-intrinsic calibration utilities."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.core.camera_calibration import (  # noqa: E402
    CameraCalibration,
    CameraCalibrationError,
    FrameUndistorter,
    calibrate_charuco_views,
    create_charuco_board,
    pixels_to_unit_rays,
)


def example_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array(
            [[900.0, 0.0, 640.0], [0.0, 880.0, 360.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.array([-0.12, 0.04, 0.001, -0.002, 0.0]),
        image_size=(1280, 720),
        rms_reprojection_error=0.31,
        n_views=20,
        camera_index=0,
        per_view_reprojection_errors=(0.2, 0.4),
    )


class CameraCalibrationPersistenceTests(unittest.TestCase):
    def test_json_round_trip_preserves_intrinsics_and_metadata(self) -> None:
        calibration = example_calibration()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            calibration.save(path)
            loaded = CameraCalibration.load(path)

        np.testing.assert_allclose(loaded.camera_matrix, calibration.camera_matrix)
        np.testing.assert_allclose(
            loaded.distortion_coefficients, calibration.distortion_coefficients
        )
        self.assertEqual(loaded.image_size, (1280, 720))
        self.assertEqual(loaded.camera_index, 0)
        self.assertEqual(loaded.per_view_reprojection_errors, (0.2, 0.4))

    def test_intrinsics_scale_only_when_aspect_ratio_matches(self) -> None:
        calibration = example_calibration()
        scaled = calibration.camera_matrix_for_size((640, 360))
        np.testing.assert_allclose(
            scaled,
            [[450.0, 0.0, 320.0], [0.0, 440.0, 180.0], [0.0, 0.0, 1.0]],
        )
        with self.assertRaisesRegex(CameraCalibrationError, "aspect ratio"):
            calibration.camera_matrix_for_size((640, 480))

    def test_undistorter_returns_a_bgr_frame_at_the_original_resolution(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :, 1] = 180
        corrected = FrameUndistorter(example_calibration()).undistort(frame)
        self.assertEqual(corrected.shape, frame.shape)
        self.assertEqual(corrected.dtype, np.uint8)

    def test_effective_post_undistort_matrix_recovers_the_physical_ray(self) -> None:
        calibration = example_calibration()
        undistorter = FrameUndistorter(calibration)
        effective_k = undistorter.camera_matrix_for_undistorted_size((1280, 720))
        # A point on a known camera ray is distorted into the raw image.  OpenCV
        # gives the corresponding coordinate in the remapped image; that point
        # must be inverted with the *effective* K, not the original raw K.
        point_camera = np.array([[0.20, -0.12, 1.0]], dtype=np.float64)
        raw_pixel, _ = cv2.projectPoints(
            point_camera,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            calibration.camera_matrix,
            calibration.distortion_coefficients,
        )
        corrected_pixel = cv2.undistortPoints(
            raw_pixel,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            P=effective_k,
        ).reshape(1, 2)
        recovered = pixels_to_unit_rays(corrected_pixel, effective_k)[0]
        expected = point_camera[0] / np.linalg.norm(point_camera[0])
        np.testing.assert_allclose(recovered, expected, atol=1e-7)


class CharucoCalibrationTests(unittest.TestCase):
    def test_synthetic_charuco_views_recover_low_reprojection_error(self) -> None:
        board = create_charuco_board()
        object_points = board.getChessboardCorners().astype(np.float32)
        ids = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)
        expected_matrix = np.array(
            [[920.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros((5, 1), dtype=np.float64)
        views: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(12):
            rvec = np.array(
                [[0.025 * ((index % 3) - 1)], [0.035 * ((index % 4) - 1.5)], [0.01 * index]]
            )
            tvec = np.array(
                [[0.025 * ((index % 4) - 1.5)], [0.02 * ((index % 3) - 1)], [0.80 + 0.03 * (index % 4)]]
            )
            image_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, expected_matrix, distortion
            )
            views.append((image_points.astype(np.float32), ids))

        calibration = calibrate_charuco_views(
            board, views, (1280, 720), camera_index=0, min_views=12
        )

        self.assertEqual(calibration.n_views, 12)
        self.assertLess(calibration.rms_reprojection_error, 0.01)
        np.testing.assert_allclose(calibration.camera_matrix, expected_matrix, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
