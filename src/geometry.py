"""Geometry helpers used by the pose analyzer."""

from __future__ import annotations

from math import acos, degrees, sqrt
from typing import Protocol


class PointLike(Protocol):
    x: float
    y: float


def joint_angle(first: PointLike, vertex: PointLike, third: PointLike) -> float:
    """Return the smaller angle (0-180 degrees) at ``vertex``."""
    vector_a = (first.x - vertex.x, first.y - vertex.y)
    vector_b = (third.x - vertex.x, third.y - vertex.y)
    length_a = sqrt(vector_a[0] ** 2 + vector_a[1] ** 2)
    length_b = sqrt(vector_b[0] ** 2 + vector_b[1] ** 2)
    if length_a == 0 or length_b == 0:
        raise ValueError("Cannot calculate an angle from overlapping points")

    cosine = (
        vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
    ) / (length_a * length_b)
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))
