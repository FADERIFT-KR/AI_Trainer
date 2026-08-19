"""Synthetic, redistribution-safe example for running the pipeline directly."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .processing import aggregate_repetitions, process_sequence
from .schema import COMMON_JOINTS
from .visualization import render_reference_preview


def _demo_depths(repetitions: int = 3, cycle_frames: int = 61) -> np.ndarray:
    standing = np.zeros(15, dtype=np.float64)
    cycle = np.sin(np.linspace(0.0, math.pi, cycle_frames, dtype=np.float64)) ** 2
    pieces = [standing]
    for index in range(repetitions):
        pieces.append(cycle)
        if index + 1 < repetitions:
            pieces.append(standing)
    pieces.append(standing)
    return np.concatenate(pieces)


def _demo_skeleton(depths: np.ndarray) -> np.ndarray:
    """Create a plausible 19-joint sequence; it is not AI Hub data."""

    frames: list[np.ndarray] = []
    for raw_depth in np.asarray(depths, dtype=np.float64):
        depth = float(raw_depth)
        hip_y = 1.00 - 0.42 * depth
        knee_y = 0.52 - 0.12 * depth
        knee_z = 0.06 + 0.32 * depth
        trunk_z = 0.18 * depth
        joints = {
            "Hip": (0.00, hip_y, 0.00),
            "LHip": (0.20, hip_y, 0.00),
            "RHip": (-0.20, hip_y, 0.00),
            "LKnee": (0.20, knee_y, knee_z),
            "RKnee": (-0.20, knee_y, knee_z),
            "LAnkle": (0.20, 0.00, 0.00),
            "RAnkle": (-0.20, 0.00, 0.00),
            "LBigToe": (0.20, -0.03, 0.25),
            "RBigToe": (-0.20, -0.03, 0.25),
            "LHeel": (0.20, 0.00, -0.10),
            "RHeel": (-0.20, 0.00, -0.10),
            "Neck": (0.00, hip_y + 0.75, trunk_z),
            "LShoulder": (0.28, hip_y + 0.72, trunk_z),
            "RShoulder": (-0.28, hip_y + 0.72, trunk_z),
            "LElbow": (0.38, hip_y + 0.48, trunk_z + 0.03),
            "RElbow": (-0.38, hip_y + 0.48, trunk_z + 0.03),
            "LWrist": (0.40, hip_y + 0.25, trunk_z + 0.10),
            "RWrist": (-0.40, hip_y + 0.25, trunk_z + 0.10),
            "Nose": (0.00, hip_y + 0.98, trunk_z + 0.04),
        }
        frames.append(np.asarray([joints[name] for name in COMMON_JOINTS], dtype=np.float64))
    return np.stack(frames)


def create_demo_preview(output_path: str | Path) -> Path:
    """Run normalization/segmentation/aggregation and save a synthetic PNG."""

    repetitions = process_sequence(
        _demo_skeleton(_demo_depths()),
        target_frames=101,
        min_flexion_deg=25.0,
        min_frames=15,
        min_valid_ratio=0.95,
        max_missing_gap_frames=2,
    )
    arrays, _, _ = aggregate_repetitions(repetitions, mad_threshold=4.0)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        render_reference_preview(
            arrays["positions_median"],
            COMMON_JOINTS,
            int(np.asarray(arrays["bottom_index"]).item()),
            handle,
            title="Synthetic air-squat reference example",
        )
    return destination
