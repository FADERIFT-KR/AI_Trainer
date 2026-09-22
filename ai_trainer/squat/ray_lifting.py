"""Camera-ray features and a temporal monocular 2D-to-3D lifting candidate.

This module implements the *input representation* used by Ray3D-style
monocular lifting: an undistorted image landmark is converted to the physical
ray leaving the calibrated camera.  A ray describes direction, not depth, so
the temporal network still has to infer root-relative 3D pose from body shape
and motion.

The project intentionally keeps this candidate separate from the live
MediaPipe-world-landmark path.  It may be promoted only after a paired,
real-webcam 3D evaluation proves an improvement.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ai_trainer.core.s1_capture.camera_calibration import CameraCalibrationError, pixels_to_unit_rays
from ai_trainer.squat.lifting_model import TemporalLiftingNet


RAY_FEATURE_TYPE = "camera_ray_v1"
PROJECTION_SCHEMA_VERSION = 1


class ProjectionCalibrationError(ValueError):
    """Raised when a 2D/3D projective-camera estimate is unusable."""


def _as_finite_array(name: str, value: np.ndarray, trailing_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < len(trailing_shape) or array.shape[-len(trailing_shape) :] != trailing_shape:
        raise ProjectionCalibrationError(
            f"{name} must have shape (..., {', '.join(map(str, trailing_shape))}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ProjectionCalibrationError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class ProjectiveCamera:
    """A pinhole camera estimated from known world-3D/image-2D correspondences.

    ``rotation_world_to_camera`` and ``translation_world_to_camera`` map a
    world point as ``X_camera = R @ X_world + t``.  The geometry is retained
    for dataset auditing; the lifting target itself is hip-relative, so the
    translation is not supplied to the neural network.
    """

    camera_matrix: np.ndarray
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    projection_matrix: np.ndarray
    fit_summary: dict[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        rotation = np.asarray(self.rotation_world_to_camera, dtype=np.float64)
        translation = np.asarray(self.translation_world_to_camera, dtype=np.float64).reshape(3)
        projection = np.asarray(self.projection_matrix, dtype=np.float64)
        if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
            raise ProjectionCalibrationError("camera_matrix must be a finite 3x3 matrix")
        if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
            raise ProjectionCalibrationError("camera_matrix must have positive focal lengths")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ProjectionCalibrationError("rotation_world_to_camera must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ProjectionCalibrationError("rotation_world_to_camera must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ProjectionCalibrationError("rotation_world_to_camera must have determinant +1")
        if not np.isfinite(translation).all():
            raise ProjectionCalibrationError("translation_world_to_camera must be finite")
        if projection.shape != (3, 4) or not np.isfinite(projection).all():
            raise ProjectionCalibrationError("projection_matrix must be a finite 3x4 matrix")
        canonical = camera_matrix @ np.column_stack((rotation, translation))
        if not np.allclose(projection, canonical, rtol=1e-5, atol=1e-5):
            raise ProjectionCalibrationError("projection_matrix is inconsistent with K, R, and t")
        object.__setattr__(self, "camera_matrix", camera_matrix)
        object.__setattr__(self, "rotation_world_to_camera", rotation)
        object.__setattr__(self, "translation_world_to_camera", translation)
        object.__setattr__(self, "projection_matrix", projection)

    def pixels_to_rays(self, pixels: np.ndarray) -> np.ndarray:
        """Return unit camera rays for undistorted, non-mirrored pixels."""
        try:
            return pixels_to_unit_rays(pixels, self.camera_matrix)
        except CameraCalibrationError as error:
            raise ProjectionCalibrationError(str(error)) from error

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        points = _as_finite_array("points_world", points_world, (3,))
        return points @ self.rotation_world_to_camera.T + self.translation_world_to_camera

    def world_relative_to_camera(self, relative_points_world: np.ndarray) -> np.ndarray:
        """Rotate Hip-relative world coordinates into camera coordinates."""
        points = _as_finite_array("relative_points_world", relative_points_world, (3,))
        return points @ self.rotation_world_to_camera.T

    def camera_relative_to_world(self, relative_points_camera: np.ndarray) -> np.ndarray:
        points = _as_finite_array("relative_points_camera", relative_points_camera, (3,))
        return points @ self.rotation_world_to_camera

    def project(self, points_world: np.ndarray) -> np.ndarray:
        points = _as_finite_array("points_world", points_world, (3,))
        homogeneous = np.concatenate(
            (points, np.ones((*points.shape[:-1], 1), dtype=np.float64)), axis=-1
        )
        projected = homogeneous @ self.projection_matrix.T
        depth = projected[..., 2:3]
        if np.any(np.abs(depth) < 1e-10):
            raise ProjectionCalibrationError("a point lies on the camera projection plane")
        return projected[..., :2] / depth

    def reprojection_errors(self, points_world: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        observed = _as_finite_array("pixels", pixels, (2,))
        predicted = self.project(points_world)
        if predicted.shape != observed.shape:
            raise ProjectionCalibrationError("points_world and pixels must have matching leading dimensions")
        return np.linalg.norm(predicted - observed, axis=-1)

    def to_dict(self) -> dict:
        return {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "feature_type": RAY_FEATURE_TYPE,
            "camera_matrix": self.camera_matrix.tolist(),
            "rotation_world_to_camera": self.rotation_world_to_camera.tolist(),
            "translation_world_to_camera": self.translation_world_to_camera.tolist(),
            "projection_matrix": self.projection_matrix.tolist(),
            "fit_summary": self.fit_summary,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProjectiveCamera":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProjectionCalibrationError(f"projective camera cannot be read: {source}") from error
        if payload.get("schema_version") != PROJECTION_SCHEMA_VERSION:
            raise ProjectionCalibrationError(
                f"unsupported projective camera schema: {payload.get('schema_version')}"
            )
        if payload.get("feature_type") != RAY_FEATURE_TYPE:
            raise ProjectionCalibrationError("projective camera feature type is incompatible")
        try:
            return cls(
                camera_matrix=np.asarray(payload["camera_matrix"]),
                rotation_world_to_camera=np.asarray(payload["rotation_world_to_camera"]),
                translation_world_to_camera=np.asarray(payload["translation_world_to_camera"]),
                projection_matrix=np.asarray(payload["projection_matrix"]),
                fit_summary=dict(payload.get("fit_summary", {})),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectionCalibrationError(f"invalid projective camera: {source}") from error


def _normalize_image_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    mean_distance = float(np.linalg.norm(points - centre, axis=1).mean())
    if mean_distance <= 1e-8:
        raise ProjectionCalibrationError("image correspondences have insufficient spatial variation")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * centre[0]], [0.0, scale, -scale * centre[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ transform.T)[:, :2], transform


def _normalize_world_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    mean_distance = float(np.linalg.norm(points - centre, axis=1).mean())
    if mean_distance <= 1e-8:
        raise ProjectionCalibrationError("world correspondences have insufficient spatial variation")
    scale = np.sqrt(3.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, 0.0, -scale * centre[0]],
            [0.0, scale, 0.0, -scale * centre[1]],
            [0.0, 0.0, scale, -scale * centre[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ transform.T)[:, :3], transform


def _normalized_dlt(world_points: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    if len(world_points) < 6:
        raise ProjectionCalibrationError("at least six 2D/3D correspondences are required")
    image_norm, image_transform = _normalize_image_points(image_points)
    world_norm, world_transform = _normalize_world_points(world_points)
    world_h = np.column_stack((world_norm, np.ones(len(world_norm))))
    matrix = np.zeros((len(world_h) * 2, 12), dtype=np.float64)
    matrix[0::2, :4] = world_h
    matrix[1::2, 4:8] = world_h
    matrix[0::2, 8:] = -image_norm[:, :1] * world_h
    matrix[1::2, 8:] = -image_norm[:, 1:] * world_h
    _singular_values, _unused, right = np.linalg.svd(matrix, full_matrices=False)
    projection_norm = right[-1].reshape(3, 4)
    return np.linalg.inv(image_transform) @ projection_norm @ world_transform


def _decompose_projection(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the projective sign which gives a conventional positive-focal K."""
    import cv2

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for sign in (1.0, -1.0):
        signed = projection * sign
        _angles, raw_k, rotation, *_unused = cv2.RQDecomp3x3(signed[:, :3])
        raw_k = np.asarray(raw_k, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        scale = float(raw_k[2, 2])
        if abs(scale) < 1e-12:
            continue
        camera_matrix = raw_k / scale
        canonical_projection = signed / scale
        if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
            continue
        translation = np.linalg.solve(camera_matrix, canonical_projection[:, 3])
        candidates.append((camera_matrix, rotation, translation, canonical_projection))
    if not candidates:
        raise ProjectionCalibrationError("could not decompose a positive-focal projective camera")
    camera_matrix, rotation, translation, _projection = candidates[0]
    canonical = camera_matrix @ np.column_stack((rotation, translation))
    return camera_matrix, rotation, translation, canonical


def estimate_projective_camera(
    world_points: np.ndarray,
    image_points: np.ndarray,
    *,
    seed: int = 20260920,
    max_initial_points: int = 30_000,
    max_refit_points: int = 80_000,
) -> ProjectiveCamera:
    """Fit a robust pinhole camera from paired 3D and 2D points.

    This estimator is useful for auditing a labelled dataset where original
    camera metadata was not shipped.  It is not a replacement for a physical
    ChArUco calibration of a user's webcam.
    """
    world = _as_finite_array("world_points", world_points, (3,)).reshape(-1, 3)
    image = _as_finite_array("image_points", image_points, (2,)).reshape(-1, 2)
    if len(world) != len(image):
        raise ProjectionCalibrationError("world_points and image_points must have equal length")
    if len(world) < 12:
        raise ProjectionCalibrationError("at least twelve 2D/3D correspondences are required")
    rng = np.random.default_rng(seed)

    def choose(count: int, limit: int) -> np.ndarray:
        return np.arange(count) if count <= limit else rng.choice(count, size=limit, replace=False)

    initial_indices = choose(len(world), max_initial_points)
    initial_projection = _normalized_dlt(world[initial_indices], image[initial_indices])
    initial_k, initial_r, initial_t, initial_canonical = _decompose_projection(initial_projection)
    initial_camera = ProjectiveCamera(initial_k, initial_r, initial_t, initial_canonical)
    initial_errors = initial_camera.reprojection_errors(world, image)
    median = float(np.median(initial_errors))
    mad = float(np.median(np.abs(initial_errors - median)))
    # The lower floor keeps normal label noise from causing an unnecessarily
    # tiny inlier set; the MAD term rejects clearly mismatched landmarks.
    inlier_threshold = max(12.0, median + 4.0 * 1.4826 * mad)
    inlier_indices = np.flatnonzero(initial_errors <= inlier_threshold)
    if len(inlier_indices) < 12:
        raise ProjectionCalibrationError("robust camera fit left too few inlier correspondences")
    if len(inlier_indices) > max_refit_points:
        inlier_indices = rng.choice(inlier_indices, size=max_refit_points, replace=False)
    projection = _normalized_dlt(world[inlier_indices], image[inlier_indices])
    camera_matrix, rotation, translation, canonical = _decompose_projection(projection)
    camera = ProjectiveCamera(camera_matrix, rotation, translation, canonical)
    errors = camera.reprojection_errors(world, image)
    summary: dict[str, float | int] = {
        "n_correspondences": int(len(world)),
        "n_initial_points": int(len(initial_indices)),
        "n_refit_inliers": int(len(inlier_indices)),
        "inlier_threshold_px": float(inlier_threshold),
        "mean_reprojection_error_px": float(errors.mean()),
        "median_reprojection_error_px": float(np.median(errors)),
        "p95_reprojection_error_px": float(np.quantile(errors, 0.95)),
    }
    return ProjectiveCamera(camera_matrix, rotation, translation, canonical, summary)


class RayTemporalLiftingNet(TemporalLiftingNet):
    """T=9 temporal lifter taking an 18-joint unit-ray sequence as input."""

    feature_type = RAY_FEATURE_TYPE
    temporal_window = 9

    def __init__(self, n_joints: int = 18, hidden: int = 128) -> None:
        super().__init__(n_joints=n_joints, hidden=hidden, input_dims=3)


__all__ = [
    "PROJECTION_SCHEMA_VERSION",
    "RAY_FEATURE_TYPE",
    "ProjectiveCamera",
    "ProjectionCalibrationError",
    "RayTemporalLiftingNet",
    "estimate_projective_camera",
]
