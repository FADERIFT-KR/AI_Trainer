"""Headless preview rendering for generated skeleton references."""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np


class VisualizationError(ValueError):
    """Raised when a reference preview cannot be rendered safely."""


SKELETON_EDGES = (
    ("Nose", "Neck"),
    ("Neck", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("Neck", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("Neck", "Hip"),
    ("Hip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),
    ("Hip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),
)


def _side_color(first: str, second: str) -> str:
    if first.startswith("L") and second.startswith("L"):
        return "#2f80ed"
    if first.startswith("R") and second.startswith("R"):
        return "#f2994a"
    return "#3c4043"


def _joint_color(name: str) -> str:
    if name.startswith("L"):
        return "#2f80ed"
    if name.startswith("R"):
        return "#f2994a"
    return "#202124"


def render_reference_preview(
    positions: np.ndarray,
    joint_names: Sequence[str],
    bottom_index: int,
    destination: BinaryIO,
    *,
    title: str = "Air-squat normalized median reference",
) -> None:
    """Render ready, bottom, and return poses as a three-panel PNG.

    Source coordinates use ``x=left/right, y=up, z=front/back``. Matplotlib's
    visual vertical is its third plotting axis, so points are displayed as
    ``(x, z, y)`` and labelled explicitly.
    """

    values = np.asarray(positions, dtype=np.float64)
    names = tuple(str(name) for name in joint_names)
    if values.ndim != 3 or values.shape[2] != 3:
        raise VisualizationError("Preview positions must have shape [frames, joints, 3]")
    if values.shape[0] < 3:
        raise VisualizationError("Preview needs at least three frames")
    if values.shape[1] != len(names):
        raise VisualizationError("Joint-name count does not match preview positions")
    if len(set(names)) != len(names):
        raise VisualizationError("Preview joint names must be unique")
    if not np.isfinite(values).all():
        raise VisualizationError("Preview positions contain NaN or infinite values")
    if not 0 <= int(bottom_index) < values.shape[0]:
        raise VisualizationError("bottom_index is outside the reference sequence")

    index = {name: position for position, name in enumerate(names)}
    missing = sorted({name for edge in SKELETON_EDGES for name in edge if name not in index})
    if missing:
        raise VisualizationError(f"Preview is missing skeleton joints: {', '.join(missing)}")

    try:
        if "MPLCONFIGDIR" not in os.environ:
            cache = Path(tempfile.gettempdir()) / "ai_trainer_matplotlib"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache)
        # Matplotlib 3.9 emits pyparsing compatibility deprecations with the
        # project's newer pinned pyparsing during import. They do not affect
        # rendering and should not pollute CLI/test output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
    except ImportError as error:  # pragma: no cover - environment dependent
        raise VisualizationError(
            "matplotlib is required for reference_preview.png; install requirements.txt "
            "or pass --no-plot"
        ) from error

    # Shared cubic limits make posture differences comparable across panels.
    displayed = values[:, :, (0, 2, 1)]
    lower = displayed.min(axis=(0, 1))
    upper = displayed.max(axis=(0, 1))
    center = (lower + upper) / 2.0
    half_range = float(np.max(upper - lower)) * 0.58
    if not np.isfinite(half_range) or half_range < 1e-6:
        raise VisualizationError("Preview coordinate range is degenerate")

    panels = (
        (0, "Ready"),
        (int(bottom_index), "Bottom"),
        (values.shape[0] - 1, "Return"),
    )
    figure = Figure(figsize=(11.5, 4.3), dpi=150, constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    for panel_index, (frame_index, panel_title) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, 3, panel_index, projection="3d")
        pose = values[frame_index]
        for first, second in SKELETON_EDGES:
            segment = pose[[index[first], index[second]]]
            axis.plot(
                segment[:, 0],
                segment[:, 2],
                segment[:, 1],
                color=_side_color(first, second),
                linewidth=2.4,
                solid_capstyle="round",
            )
        colors = [_joint_color(name) for name in names]
        axis.scatter(
            pose[:, 0],
            pose[:, 2],
            pose[:, 1],
            c=colors,
            s=20,
            depthshade=False,
            edgecolors="white",
            linewidths=0.35,
        )
        axis.set_xlim(center[0] - half_range, center[0] + half_range)
        axis.set_ylim(center[1] - half_range, center[1] + half_range)
        axis.set_zlim(center[2] - half_range, center[2] + half_range)
        axis.set_box_aspect((1, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(elev=17, azim=-68)
        axis.set_xlabel("x: left/right", fontsize=8, labelpad=2)
        axis.set_ylabel("z: front/back", fontsize=8, labelpad=2)
        axis.set_zlabel("")
        axis.text2D(
            0.86,
            0.54,
            "y: up",
            transform=axis.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
        )
        axis.tick_params(labelsize=6, pad=0)
        axis.set_title(f"{panel_title} · frame {frame_index}", fontsize=10, pad=4)
        axis.grid(True, alpha=0.28)

    figure.suptitle(title, fontsize=13)
    canvas.print_png(destination)
    figure.clear()
