"""Small, type-safe adapter for text passed to OpenCV drawing calls."""
from __future__ import annotations

from enum import Enum

import cv2

_WARNED_SOURCES: set[str] = set()


def display_text(value: object | None, *, source: str = "opencv_text") -> str:
    """Return user-facing text without exposing an object's debug representation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        candidate = value.value if isinstance(value.value, str) else value.name
        return candidate
    for attribute in ("message", "label"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, str):
            return candidate
    print(
        f"[PUTTEXT ERROR] source={source} value={value!r} type={type(value).__name__}",
        flush=True,
    )
    return ""


def safe_put_text(
    frame,
    text,
    org,
    font_face,
    font_scale,
    color,
    thickness,
    line_type=cv2.LINE_AA,
    *,
    source: str = "opencv_text",
):
    """Draw one label without allowing an annotation failure to stop analysis."""
    normalized = display_text(text, source=source)
    if not normalized:
        return frame
    try:
        cv2.putText(
            frame, normalized, org, font_face, font_scale, color, thickness, line_type
        )
    except Exception as error:  # OpenCV bindings can raise cv2.error or TypeError.
        if source not in _WARNED_SOURCES:
            _WARNED_SOURCES.add(source)
            print(
                f"[UI TEXT WARNING] source={source} type={type(text).__name__} "
                f"value={text!r} error={error}",
                flush=True,
            )
    return frame
