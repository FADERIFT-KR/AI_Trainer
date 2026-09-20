"""Single-camera intrinsic calibration and low-overhead frame undistortion.

The live workout pipeline uses MediaPipe on a single RGB camera.  A per-camera
ChArUco calibration estimates the camera matrix and lens-distortion
coefficients once, then undistorts every raw frame *before* mirroring and pose
inference.  It does not estimate the camera's pose in the room; the existing
``OnlineSquatSession`` standing-pose calibration remains responsible for
per-session body scale and orientation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
ASPECT_RATIO_TOLERANCE = 0.01


class CameraCalibrationError(ValueError):
    """Raised when calibration data is invalid or does not match a camera frame."""


def pixels_to_unit_rays(pixels: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Convert undistorted pixel coordinates into unit camera rays.

    ``pixels`` may have any leading dimensions as long as its last dimension is
    ``(u, v)``.  The supplied matrix must describe *the image on which those
    pixels were detected*.  In particular, callers that run inference after
    :class:`FrameUndistorter` must use its effective, post-undistortion matrix
    rather than the original raw-sensor matrix.

    The result is expressed in the camera coordinate system.  It is a direction
    only; a single RGB camera cannot infer a point's distance along this ray.
    """
    points = np.asarray(pixels, dtype=np.float64)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise CameraCalibrationError(
            f"pixels must have shape (..., 2), got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise CameraCalibrationError("pixels must contain only finite values")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise CameraCalibrationError("camera_matrix must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or abs(matrix[2, 2]) < 1e-12:
        raise CameraCalibrationError("camera_matrix must have positive focal lengths")
    if np.linalg.cond(matrix) > 1e12:
        raise CameraCalibrationError("camera_matrix is singular or ill-conditioned")

    homogeneous = np.concatenate(
        (points, np.ones((*points.shape[:-1], 1), dtype=np.float64)), axis=-1
    )
    # The coordinates are represented as row vectors here, hence ``inv(K).T``.
    rays = homogeneous @ np.linalg.inv(matrix).T
    norm = np.linalg.norm(rays, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12) or not np.isfinite(norm).all():
        raise CameraCalibrationError("camera rays could not be normalized")
    return rays / norm


def unmirror_pixels(pixels: np.ndarray, image_width: int) -> np.ndarray:
    """Map selfie-mirrored pixel coordinates back to physical camera pixels."""
    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise CameraCalibrationError(
            f"pixels must have shape (..., 2), got {points.shape}"
        )
    if image_width <= 0:
        raise CameraCalibrationError("image_width must be positive")
    result = points.copy()
    result[..., 0] = float(image_width - 1) - result[..., 0]
    return result


@dataclass(frozen=True)
class CameraCalibration:
    """Validated intrinsic parameters for one camera and capture resolution."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int]  # (width, height)
    rms_reprojection_error: float
    n_views: int
    camera_index: int | None = None
    per_view_reprojection_errors: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(-1, 1)
        width, height = self.image_size
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise CameraCalibrationError("camera_matrix must be a finite 3x3 matrix")
        if distortion.size < 4 or not np.isfinite(distortion).all():
            raise CameraCalibrationError("distortion_coefficients must contain at least four finite values")
        if width <= 0 or height <= 0:
            raise CameraCalibrationError("image_size must contain positive width and height")
        if self.n_views < 1 or not np.isfinite(self.rms_reprojection_error) or self.rms_reprojection_error < 0:
            raise CameraCalibrationError("n_views and rms_reprojection_error are invalid")
        if self.camera_index is not None and self.camera_index < 0:
            raise CameraCalibrationError("camera_index cannot be negative")
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)
        object.__setattr__(self, "image_size", (int(width), int(height)))
        object.__setattr__(
            self,
            "per_view_reprojection_errors",
            tuple(float(value) for value in self.per_view_reprojection_errors),
        )

    def camera_matrix_for_size(self, image_size: tuple[int, int]) -> np.ndarray:
        """Scale intrinsics for an equal-aspect-ratio capture resolution."""
        width, height = image_size
        base_width, base_height = self.image_size
        if width <= 0 or height <= 0:
            raise CameraCalibrationError("frame size must be positive")
        base_aspect = base_width / base_height
        frame_aspect = width / height
        if abs(frame_aspect - base_aspect) / base_aspect > ASPECT_RATIO_TOLERANCE:
            raise CameraCalibrationError(
                "calibration aspect ratio does not match the camera frame "
                f"({base_width}x{base_height} vs {width}x{height})"
            )
        scale = np.diag([width / base_width, height / base_height, 1.0])
        return scale @ self.camera_matrix

    def require_camera_index(self, camera_index: int) -> None:
        if self.camera_index is not None and self.camera_index != camera_index:
            raise CameraCalibrationError(
                f"calibration was recorded for camera {self.camera_index}, not camera {camera_index}"
            )

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "camera_index": self.camera_index,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.reshape(-1).tolist(),
            "rms_reprojection_error": self.rms_reprojection_error,
            "n_views": self.n_views,
            "per_view_reprojection_errors": list(self.per_view_reprojection_errors),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CameraCalibrationError(f"camera calibration cannot be read: {path}") from error
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise CameraCalibrationError(f"unsupported camera calibration schema: {payload.get('schema_version')}")
        size = payload.get("image_size", {})
        try:
            return cls(
                camera_matrix=np.asarray(payload["camera_matrix"]),
                distortion_coefficients=np.asarray(payload["distortion_coefficients"]),
                image_size=(int(size["width"]), int(size["height"])),
                rms_reprojection_error=float(payload["rms_reprojection_error"]),
                n_views=int(payload["n_views"]),
                camera_index=payload.get("camera_index"),
                per_view_reprojection_errors=tuple(payload.get("per_view_reprojection_errors", [])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CameraCalibrationError(f"invalid camera calibration data: {path}") from error


class FrameUndistorter:
    """Caches OpenCV remap tables and undistorts raw BGR frames at capture rate."""

    def __init__(self, calibration: CameraCalibration, *, alpha: float = 0.0) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.calibration = calibration
        self.alpha = alpha
        # ``new_matrix`` is deliberately retained: pose landmarks are inferred
        # on this remapped image, so a later ray conversion must use this exact
        # effective K rather than the raw-sensor K.
        self._maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _maps_for_size(self, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import cv2

        width, height = image_size
        if width <= 0 or height <= 0:
            raise CameraCalibrationError("frame size must be positive")
        maps = self._maps.get(image_size)
        if maps is None:
            camera_matrix = self.calibration.camera_matrix_for_size(image_size)
            new_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                self.calibration.distortion_coefficients,
                image_size,
                self.alpha,
                image_size,
            )
            map_x, map_y = cv2.initUndistortRectifyMap(
                camera_matrix,
                self.calibration.distortion_coefficients,
                None,
                new_matrix,
                image_size,
                cv2.CV_16SC2,
            )
            maps = (map_x, map_y, np.asarray(new_matrix, dtype=np.float64))
            self._maps[image_size] = maps
        return maps

    def camera_matrix_for_undistorted_size(self, image_size: tuple[int, int]) -> np.ndarray:
        """Return the effective K used for a remapped output image.

        This initializes the remap cache if necessary.  Returning a copy keeps
        caller-side feature processing from mutating the calibration cache.
        """
        return self._maps_for_size(image_size)[2].copy()

    def undistort(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("camera frame must be uint8 BGR [H, W, 3]")
        height, width = frame.shape[:2]
        maps = self._maps_for_size((width, height))
        return cv2.remap(frame, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)


def create_charuco_board(
    *,
    squares_x: int = 5,
    squares_y: int = 7,
    square_length_m: float = 0.03,
    marker_length_m: float = 0.022,
    dictionary_id: int | None = None,
):
    """Create the ChArUco target used by the calibration capture script."""
    import cv2

    if squares_x < 3 or squares_y < 3:
        raise ValueError("ChArUco board needs at least 3 squares in each direction")
    if not 0 < marker_length_m < square_length_m:
        raise ValueError("marker_length_m must be positive and smaller than square_length_m")
    if dictionary_id is None:
        dictionary_id = cv2.aruco.DICT_5X5_100
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_length_m, marker_length_m, dictionary)


def calibrate_charuco_views(
    board,
    views: list[tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
    *,
    camera_index: int | None = None,
    min_views: int = 12,
) -> CameraCalibration:
    """Estimate intrinsics from detected ChArUco corners and reject bad views once."""
    import cv2

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for corners, ids in views:
        if corners is None or ids is None or len(ids) < 4:
            continue
        obj, image = board.matchImagePoints(corners, ids)
        if len(obj) >= 4:
            object_points.append(obj)
            image_points.append(image)
    if len(object_points) < min_views:
        raise CameraCalibrationError(
            f"at least {min_views} valid ChArUco views are required; received {len(object_points)}"
        )

    def fit(points_3d: list[np.ndarray], points_2d: list[np.ndarray]):
        return cv2.calibrateCamera(points_3d, points_2d, image_size, None, None)

    rms, matrix, distortion, rvecs, tvecs = fit(object_points, image_points)
    view_errors = _per_view_reprojection_errors(
        object_points, image_points, rvecs, tvecs, matrix, distortion
    )
    median_error = float(np.median(view_errors))
    max_error = max(1.0, median_error * 2.5)
    keep = [error <= max_error for error in view_errors]
    if not all(keep) and sum(keep) >= min_views:
        object_points = [points for points, accepted in zip(object_points, keep) if accepted]
        image_points = [points for points, accepted in zip(image_points, keep) if accepted]
        rms, matrix, distortion, rvecs, tvecs = fit(object_points, image_points)
        view_errors = _per_view_reprojection_errors(
            object_points, image_points, rvecs, tvecs, matrix, distortion
        )

    return CameraCalibration(
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        image_size=image_size,
        rms_reprojection_error=float(rms),
        n_views=len(object_points),
        camera_index=camera_index,
        per_view_reprojection_errors=tuple(view_errors),
    )


def _per_view_reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    import cv2

    errors = []
    for points_3d, points_2d, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(points_3d, rvec, tvec, matrix, distortion)
        delta = projected.reshape(-1, 2) - points_2d.reshape(-1, 2)
        errors.append(float(np.linalg.norm(delta, axis=1).mean()))
    return errors


__all__ = [
    "CameraCalibration",
    "CameraCalibrationError",
    "FrameUndistorter",
    "calibrate_charuco_views",
    "create_charuco_board",
    "pixels_to_unit_rays",
    "unmirror_pixels",
]
