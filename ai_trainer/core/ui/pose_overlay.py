"""Fast NumPy raster rendering for 2-D and projected 3-D pose views."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


# Canonical 33-landmark MediaPipe Pose graph.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)

LEFT_LANDMARKS = frozenset({1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31})
RIGHT_LANDMARKS = frozenset({4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32})


def _draw_line(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
    xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int32)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int32)
    radius = max(0, thickness // 2)
    height, width = canvas.shape[:2]
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            draw_x = xs + offset_x
            draw_y = ys + offset_y
            valid = (draw_x >= 0) & (draw_x < width) & (draw_y >= 0) & (draw_y < height)
            canvas[draw_y[valid], draw_x[valid]] = color


def _draw_circle(
    canvas: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    center_x, center_y = center
    height, width = canvas.shape[:2]
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    canvas[y0:y1, x0:x1][mask] = color


def _valid_landmarks(points: np.ndarray, visibility_threshold: float) -> np.ndarray:
    valid = np.all(np.isfinite(points[:, :3]), axis=1)
    if points.shape[1] >= 4:
        valid &= np.isfinite(points[:, 3]) & (points[:, 3] >= visibility_threshold)
    return valid


def _limb_color(start: int, end: int) -> tuple[int, int, int]:
    if start in LEFT_LANDMARKS and end in LEFT_LANDMARKS:
        return (105, 235, 120)  # BGR green
    if start in RIGHT_LANDMARKS and end in RIGHT_LANDMARKS:
        return (80, 165, 255)  # BGR orange
    return (255, 220, 80)  # BGR cyan


def draw_2d_pose(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    *,
    visibility_threshold: float = 0.45,
) -> np.ndarray:
    """Draw normalized MediaPipe landmarks over a BGR camera frame."""

    canvas = np.ascontiguousarray(frame_bgr.copy())
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"landmarks must have shape [N, >=3], got {points.shape}")
    valid = _valid_landmarks(points, visibility_threshold)
    height, width = canvas.shape[:2]
    pixel_points = np.zeros((len(points), 2), dtype=np.int32)
    pixel_points[valid, 0] = np.rint(points[valid, 0] * (width - 1)).astype(np.int32)
    pixel_points[valid, 1] = np.rint(points[valid, 1] * (height - 1)).astype(np.int32)

    for start, end in POSE_CONNECTIONS:
        if start >= len(points) or end >= len(points) or not (valid[start] and valid[end]):
            continue
        _draw_line(
            canvas,
            tuple(pixel_points[start]),
            tuple(pixel_points[end]),
            _limb_color(start, end),
            thickness=3,
        )
    for index, pixel in enumerate(pixel_points):
        if valid[index]:
            _draw_circle(canvas, tuple(pixel), 4, (245, 245, 245))
            _draw_circle(canvas, tuple(pixel), 2, _limb_color(index, index))
    return canvas


def _draw_grid(canvas: np.ndarray) -> None:
    height, width = canvas.shape[:2]
    color = (48, 55, 68)
    horizon = int(height * 0.82)
    for fraction in np.linspace(0.12, 0.88, 7):
        x = int(width * fraction)
        _draw_line(canvas, (width // 2, int(height * 0.55)), (x, horizon), color, thickness=1)
    for fraction in np.linspace(0.60, 0.84, 5):
        y = int(height * fraction)
        inset = int((y - height * 0.55) * 0.55)
        _draw_line(canvas, (inset, y), (width - inset, y), color, thickness=1)


def _project_world_landmarks(
    points: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    # MediaPipe world: x right, y down, z away/forward convention varies by
    # view.  Display convention is x right, y up, z toward the viewer.
    coordinates = points[:, :3].astype(np.float64, copy=True)
    coordinates[:, 1] *= -1.0
    coordinates[:, 2] *= -1.0

    if len(points) > 24 and valid[23] and valid[24]:
        origin = (coordinates[23] + coordinates[24]) * 0.5
    else:
        origin = np.nanmedian(coordinates[valid], axis=0)
    coordinates -= origin

    yaw = math.radians(-24.0)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    view_x = coordinates[:, 0] * cos_yaw + coordinates[:, 2] * sin_yaw
    view_depth = -coordinates[:, 0] * sin_yaw + coordinates[:, 2] * cos_yaw
    view_y = coordinates[:, 1] - 0.18 * view_depth

    extent_x = float(np.ptp(view_x[valid]))
    extent_y = float(np.ptp(view_y[valid]))
    scale = min(
        width * 0.72 / max(extent_x, 0.25),
        height * 0.78 / max(extent_y, 0.40),
    )
    screen = np.zeros((len(points), 2), dtype=np.int32)
    screen[valid, 0] = np.rint(width * 0.5 + view_x[valid] * scale).astype(np.int32)
    screen[valid, 1] = np.rint(height * 0.50 - view_y[valid] * scale).astype(np.int32)
    return screen, view_depth


def render_3d_pose(
    landmarks: np.ndarray | None,
    *,
    width: int = 640,
    height: int = 480,
    visibility_threshold: float = 0.45,
) -> np.ndarray:
    """Render world landmarks as a responsive three-quarter 3-D BGR view."""

    if width < 64 or height < 64:
        raise ValueError("3-D canvas must be at least 64 x 64 pixels")
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = (28, 32, 42)
    _draw_grid(canvas)
    if landmarks is None:
        return np.ascontiguousarray(canvas)

    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"landmarks must have shape [N, >=3], got {points.shape}")
    valid = _valid_landmarks(points, visibility_threshold)
    if np.count_nonzero(valid) < 2:
        return np.ascontiguousarray(canvas)
    screen, depth = _project_world_landmarks(points, valid, width, height)

    visible_edges: list[tuple[float, int, int]] = []
    for start, end in POSE_CONNECTIONS:
        if start < len(points) and end < len(points) and valid[start] and valid[end]:
            visible_edges.append(((depth[start] + depth[end]) * 0.5, start, end))
    for _, start, end in sorted(visible_edges, reverse=True):
        _draw_line(
            canvas,
            tuple(screen[start]),
            tuple(screen[end]),
            _limb_color(start, end),
            thickness=4,
        )
    for index, pixel in enumerate(screen):
        if valid[index]:
            _draw_circle(canvas, tuple(pixel), 5, (238, 242, 248))
            _draw_circle(canvas, tuple(pixel), 3, _limb_color(index, index))

    # Small axis triad communicates that this is a projected 3-D view.
    axis_origin = (44, height - 40)
    _draw_line(canvas, axis_origin, (82, height - 40), (80, 110, 255), thickness=3)
    _draw_line(canvas, axis_origin, (44, height - 82), (105, 235, 120), thickness=3)
    _draw_line(canvas, axis_origin, (68, height - 60), (255, 190, 80), thickness=3)
    return np.ascontiguousarray(canvas)


__all__ = ["POSE_CONNECTIONS", "draw_2d_pose", "render_3d_pose"]
