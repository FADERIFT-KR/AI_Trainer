"""Headless tests for the real-time webcam pose pipeline.

The tests deliberately avoid opening a camera or GUI window.  MediaPipe is
represented by small fake landmark/detector objects, so the pure conversion
and frame-processing contracts remain testable on CI machines.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_trainer.live_pose.core import FrameProcessor, PoseObservation, landmarks_to_array
from ai_trainer.live_pose.render import render_3d_pose
from scripts.download_pose_model import DEFAULT_OUTPUT, _valid_model


def fake_landmarks() -> SimpleNamespace:
    """Return a MediaPipe-style 33-landmark container with known values."""

    return SimpleNamespace(
        landmark=[
            SimpleNamespace(
                x=index / 100.0,
                y=(index + 1) / 100.0,
                z=-index / 200.0,
                visibility=0.5 + index / 100.0,
            )
            for index in range(33)
        ]
    )


class FakeDetector:
    """Detector double that records its RGB input and returns one pose."""

    def __init__(self, observation: PoseObservation | None) -> None:
        self.observation = observation
        self.inputs: list[np.ndarray] = []

    def process(self, frame_rgb: np.ndarray) -> PoseObservation:
        self.inputs.append(frame_rgb.copy())
        return self.observation


class LandmarkConversionTests(unittest.TestCase):
    def test_mediapipe_landmarks_are_converted_to_float32_xyzw_array(self) -> None:
        converted = landmarks_to_array(fake_landmarks())

        self.assertEqual(converted.shape, (33, 4))
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_allclose(converted[0], [0.0, 0.01, 0.0, 0.5])
        np.testing.assert_allclose(converted[32], [0.32, 0.33, -0.16, 0.82])

    def test_tasks_api_iterable_and_optional_visibility_are_supported(self) -> None:
        landmarks = fake_landmarks().landmark
        landmarks[4].visibility = None

        converted = landmarks_to_array(landmarks)

        self.assertEqual(converted.shape, (33, 4))
        self.assertEqual(converted[4, 3], 1.0)

    def test_missing_landmarks_are_preserved_as_no_detection(self) -> None:
        self.assertIsNone(landmarks_to_array(None))


class FrameProcessorTests(unittest.TestCase):
    def test_fake_detector_receives_mirrored_rgb_and_outputs_both_views(self) -> None:
        image_points = landmarks_to_array(fake_landmarks())
        world_points = image_points.copy()
        detector = FakeDetector(PoseObservation(image_points, world_points))
        processor = FrameProcessor(detector, mirror=True)

        # Asymmetric BGR pixels make both horizontal mirroring and channel
        # conversion observable without requiring a real camera.
        frame_bgr = np.zeros((4, 6, 3), dtype=np.uint8)
        frame_bgr[:, :, 0] = np.arange(6, dtype=np.uint8)
        frame_bgr[:, :, 1] = 30
        frame_bgr[:, :, 2] = 200
        expected_video = np.ascontiguousarray(frame_bgr[:, ::-1])
        expected_rgb = np.ascontiguousarray(expected_video[:, :, ::-1])
        skeleton = np.full((12, 16, 3), 77, dtype=np.uint8)

        # Rendering is tested separately.  Replacing it here keeps this unit
        # test independent of OpenCV and verifies processor orchestration.
        with (
            patch(
                "ai_trainer.live_pose.core.draw_2d_pose",
                side_effect=lambda frame, points: frame.copy(),
            ) as draw_2d,
            patch(
                "ai_trainer.live_pose.core.render_3d_pose",
                return_value=skeleton,
            ) as draw_3d,
        ):
            processed = processor.process(frame_bgr)

        self.assertTrue(processed.pose_found)
        np.testing.assert_array_equal(detector.inputs[0], expected_rgb)
        np.testing.assert_array_equal(processed.video_bgr, expected_video)
        np.testing.assert_array_equal(processed.skeleton_bgr, skeleton)
        draw_2d.assert_called_once()
        np.testing.assert_array_equal(draw_2d.call_args.args[1], image_points)
        draw_3d.assert_called_once()
        np.testing.assert_array_equal(draw_3d.call_args.args[0], world_points)

    def test_no_detection_clears_previous_skeleton_and_sets_pose_found_false(self) -> None:
        points = landmarks_to_array(fake_landmarks())
        detector = FakeDetector(PoseObservation(points, points.copy()))
        processor = FrameProcessor(
            detector,
            mirror=False,
            skeleton_width=128,
            skeleton_height=96,
        )
        frame_bgr = np.zeros((48, 64, 3), dtype=np.uint8)

        detected = processor.process(frame_bgr)
        detector.observation = None
        missing = processor.process(frame_bgr)

        self.assertTrue(detected.pose_found)
        self.assertFalse(missing.pose_found)
        self.assertEqual(missing.skeleton_bgr.shape, (96, 128, 3))
        np.testing.assert_array_equal(
            missing.skeleton_bgr,
            render_3d_pose(None, width=128, height=96),
        )
        self.assertFalse(np.array_equal(detected.skeleton_bgr, missing.skeleton_bgr))
        # The clean camera frame also proves that a prior 2-D overlay is not
        # retained when the detector loses the person.
        np.testing.assert_array_equal(missing.video_bgr, frame_bgr)

    def test_invalid_camera_frame_shape_and_dtype_are_rejected_before_inference(self) -> None:
        detector = FakeDetector(None)
        processor = FrameProcessor(detector)
        invalid_frames = (
            np.zeros((12, 16), dtype=np.uint8),
            np.zeros((12, 16, 4), dtype=np.uint8),
            np.zeros((12, 16, 3), dtype=np.float32),
        )

        for invalid in invalid_frames:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaisesRegex(ValueError, "uint8 BGR"):
                    processor.process(invalid)

        self.assertEqual(detector.inputs, [])


class SkeletonRenderingTests(unittest.TestCase):
    def test_render_3d_pose_returns_nonempty_uint8_bgr_canvas(self) -> None:
        points = landmarks_to_array(fake_landmarks())

        try:
            image = render_3d_pose(points, width=192, height=128)
        except RuntimeError as error:
            if "opencv" in str(error).casefold() or "cv2" in str(error).casefold():
                self.skipTest(f"OpenCV is not installed in this test environment: {error}")
            raise

        self.assertEqual(image.shape, (128, 192, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(image)), 0)
        self.assertTrue(image.flags.c_contiguous)


class PoseModelBundleTests(unittest.TestCase):
    def test_downloaded_model_passes_offline_integrity_and_bundle_validation(self) -> None:
        if not DEFAULT_OUTPUT.is_file():
            self.skipTest("The optional MediaPipe pose model has not been downloaded")

        self.assertTrue(_valid_model(DEFAULT_OUTPUT))


if __name__ == "__main__":
    unittest.main()
