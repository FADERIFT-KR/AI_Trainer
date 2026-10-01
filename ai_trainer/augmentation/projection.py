"""Simple auditable 3-D to 2-D view projection for skeleton-only datasets."""
from __future__ import annotations

import numpy as np


def _rotation_x(degrees: float) -> np.ndarray:
    value = np.radians(degrees)
    c, s = np.cos(value), np.sin(value)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(degrees: float) -> np.ndarray:
    value = np.radians(degrees)
    c, s = np.cos(value), np.sin(value)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def orthographic_project(
    coords_3d: np.ndarray,
    *,
    yaw_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
    scale: float = 1.0,
    image_center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Project body-aligned coordinates; yaw 15-45 degrees represents oblique views."""
    coords = np.asarray(coords_3d, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords_3d must have shape (T,J,3)")
    rotation = _rotation_x(pitch_degrees) @ _rotation_y(yaw_degrees)
    viewed = np.einsum("ij,tpj->tpi", rotation, coords)
    output = np.stack([viewed[..., 0], -viewed[..., 1]], axis=-1) * scale
    return output + np.asarray(image_center, dtype=np.float64)
