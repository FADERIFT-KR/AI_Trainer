"""Phase-aligned fusion of sequential front/left/right squat recordings.

The application currently records the three views with one camera, one after
another.  They are therefore *not* simultaneous observations and must not be
triangulated as if they were.  This module pairs repetitions by their index,
resamples each repetition by normalized motion progress, and produces a
canonical feedback skeleton:

* front view is trusted most for anatomical lateral (x) and vertical (y);
* side views are trusted most for sagittal depth (z) and vertical (y);
* MediaPipe visibility, the live freeze mask, and near/far-side occlusion are
  reflected in per-joint weights;
* the resulting trajectory is refined offline to remove spikes and jitter.

The output is intended for post-session visual feedback.  It is deliberately
kept separate from the per-view DTW/ML decision so sequential recordings do
not masquerade as calibrated synchronous multi-camera ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .camera_views import VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT, VIEWS
from .common_skeleton import COMMON_JOINT_NAMES
from .skeleton_filter import SkeletonFilterConfig, SkeletonRefinementReport, refine_skeleton_sequence


_JOINT_COUNT = len(COMMON_JOINT_NAMES)
_IDX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
_MP_DIRECT = {
    "LShoulder": 11, "RShoulder": 12,
    "LElbow": 13, "RElbow": 14,
    "LWrist": 15, "RWrist": 16,
    "LHip": 23, "RHip": 24,
    "LKnee": 25, "RKnee": 26,
    "LAnkle": 27, "RAnkle": 28,
    "LHeel": 29, "RHeel": 30,
    "LBigToe": 31, "RBigToe": 32,
}


@dataclass(frozen=True)
class FusedRepetition:
    """One canonical repetition built from phase-aligned recorded views."""

    rep_index: int
    coordinates: np.ndarray  # (T, 18, 3), canonical body coordinates
    source_views: tuple[str, ...]
    refinement: SkeletonRefinementReport | None


@dataclass(frozen=True)
class MultiViewFusionResult:
    repetitions: tuple[FusedRepetition, ...]
    frame_poses: dict[str, tuple[np.ndarray | None, ...]]
    frame_sources: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _Segment:
    row_indices: np.ndarray
    analysis_frames: np.ndarray
    coordinates: np.ndarray
    confidence: np.ndarray
    frozen: np.ndarray


def _valid_pose(row: dict) -> np.ndarray | None:
    # display_aligned_3d contains the side-view image-plane correction that is
    # specifically intended to fix visually implausible monocular depth.  It
    # does not feed back into classification, and is appropriate for this
    # post-session feedback-only product.
    for key in ("display_aligned_3d", "aligned_3d"):
        value = row.get(key)
        if value is None:
            continue
        pose = np.asarray(value, dtype=float)
        if pose.shape == (_JOINT_COUNT, 3) and np.isfinite(pose).all():
            return pose
    return None


def _common_visibility(row: dict) -> np.ndarray:
    landmarks = np.asarray(row.get("world_landmarks"), dtype=float)
    if landmarks.shape != (33, 4):
        return np.ones(_JOINT_COUNT, dtype=float)
    visibility = np.clip(landmarks[:, 3], 0.0, 1.0)
    result = np.ones(_JOINT_COUNT, dtype=float)
    for name, mp_index in _MP_DIRECT.items():
        result[_IDX[name]] = visibility[mp_index]
    result[_IDX["Hip"]] = min(visibility[23], visibility[24])
    result[_IDX["Neck"]] = min(visibility[11], visibility[12])
    return result


def _segment_rows(rows: Sequence[dict]) -> list[_Segment]:
    events = [row for row in rows if row.get("completed_rep") is not None]
    segments: list[_Segment] = []
    for event in events:
        frame_range = event["completed_rep"].get("frame_range")
        if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
            continue
        start, end = int(frame_range[0]), int(frame_range[1])
        selected: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for row_index, row in enumerate(rows):
            analysis_frame = row.get("analysis_frame")
            if not isinstance(analysis_frame, int) or not start <= analysis_frame <= end:
                continue
            pose = _valid_pose(row)
            if pose is None:
                continue
            frozen = np.asarray(row.get("frozen_3d", np.zeros(_JOINT_COUNT)), dtype=bool)
            if frozen.shape != (_JOINT_COUNT,):
                frozen = np.zeros(_JOINT_COUNT, dtype=bool)
            selected.append((row_index, analysis_frame, pose, _common_visibility(row), frozen))
        if len(selected) < 2:
            continue
        segments.append(_Segment(
            row_indices=np.asarray([item[0] for item in selected], dtype=int),
            analysis_frames=np.asarray([item[1] for item in selected], dtype=float),
            coordinates=np.stack([item[2] for item in selected]),
            confidence=np.stack([item[3] for item in selected]),
            frozen=np.stack([item[4] for item in selected]),
        ))
    return segments


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == length:
        return values.copy()
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, length)
    flat = values.reshape(len(values), -1)
    output = np.empty((length, flat.shape[1]), dtype=float)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(target, source, flat[:, column])
    return output.reshape((length,) + values.shape[1:])


def _view_coordinate_weights(view: str, confidence: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    if view == VIEW_FRONT:
        axis_weights = np.array([1.0, 0.90, 0.15])
    else:
        axis_weights = np.array([0.15, 0.75, 1.0])
    reliability = np.clip(confidence, 0.05, 1.0) ** 2
    reliability = np.where(frozen, reliability * 0.08, reliability)

    if view in (VIEW_LEFT, VIEW_RIGHT):
        near_prefix = "L" if view == VIEW_LEFT else "R"
        far_prefix = "R" if view == VIEW_LEFT else "L"
        for joint_index, name in enumerate(COMMON_JOINT_NAMES):
            if name.startswith(near_prefix):
                reliability[:, joint_index] *= 1.15
            elif name.startswith(far_prefix):
                reliability[:, joint_index] *= 0.30
    return reliability[..., None] * axis_weights


def _align_side_depth(resampled: dict[str, np.ndarray]) -> None:
    """Resolve the possible z sign ambiguity between opposite side captures."""
    if VIEW_LEFT not in resampled or VIEW_RIGHT not in resampled:
        return
    left_z = resampled[VIEW_LEFT][..., 2]
    right_z = resampled[VIEW_RIGHT][..., 2]
    direct = float(np.mean((left_z - right_z) ** 2))
    flipped = float(np.mean((left_z + right_z) ** 2))
    if flipped < direct:
        resampled[VIEW_RIGHT][..., 2] *= -1.0


def _fuse_repetition(rep_index: int, segments: Mapping[str, _Segment], fps: float) -> FusedRepetition:
    length = max(len(segment.coordinates) for segment in segments.values())
    resampled = {view: _resample(segment.coordinates, length) for view, segment in segments.items()}
    confidence = {view: _resample(segment.confidence, length) for view, segment in segments.items()}
    frozen = {
        view: _resample(segment.frozen.astype(float), length) >= 0.5
        for view, segment in segments.items()
    }
    _align_side_depth(resampled)

    numerator = np.zeros((length, _JOINT_COUNT, 3), dtype=float)
    denominator = np.zeros_like(numerator)
    for view in segments:
        weights = _view_coordinate_weights(view, confidence[view], frozen[view])
        numerator += resampled[view] * weights
        denominator += weights
    fused = numerator / np.maximum(denominator, 1e-8)
    # Enforce the documented spatial-normalization contract after blending.
    fused -= fused[:, _IDX["Hip"]:_IDX["Hip"] + 1, :]

    report: SkeletonRefinementReport | None = None
    if length >= 5:
        safe_fps = max(float(fps), 6.0)
        cutoff = min(4.0, safe_fps * 0.40)
        fused, report = refine_skeleton_sequence(
            fused,
            COMMON_JOINT_NAMES,
            SkeletonFilterConfig(
                fps=safe_fps,
                cutoff_hz=cutoff,
                bone_length_strength=0.45,
            ),
        )
        fused -= fused[:, _IDX["Hip"]:_IDX["Hip"] + 1, :]
    return FusedRepetition(rep_index, fused, tuple(view for view in VIEWS if view in segments), report)


def _pose_at_progress(track: np.ndarray, progress: float) -> np.ndarray:
    position = float(np.clip(progress, 0.0, 1.0)) * (len(track) - 1)
    low = int(np.floor(position))
    high = min(low + 1, len(track) - 1)
    alpha = position - low
    return (1.0 - alpha) * track[low] + alpha * track[high]


def fuse_recorded_views(
    rows_by_view: Mapping[str, Sequence[dict]],
    fps_by_view: Mapping[str, float] | None = None,
) -> MultiViewFusionResult:
    """Fuse matched repetitions and map them back onto every replay timeline.

    Frames outside a matched repetition retain their current view's normalized
    skeleton and are labelled ``single_view``.  This keeps preparation footage
    visible without falsely claiming that unmatched instants were fused.
    """
    segments_by_view = {view: _segment_rows(rows) for view, rows in rows_by_view.items()}
    rep_count = max((len(segments) for segments in segments_by_view.values()), default=0)
    fps_values = [float(value) for value in (fps_by_view or {}).values() if float(value) > 0]
    fps = float(np.median(fps_values)) if fps_values else 30.0

    repetitions: list[FusedRepetition] = []
    for rep_index in range(rep_count):
        available = {
            view: segments[rep_index]
            for view, segments in segments_by_view.items()
            if rep_index < len(segments)
        }
        # A comprehensive result needs the frontal plane and at least one
        # sagittal view.  Otherwise retain per-view output only.
        if VIEW_FRONT not in available or not ({VIEW_LEFT, VIEW_RIGHT} & available.keys()):
            continue
        repetitions.append(_fuse_repetition(rep_index, available, fps))

    fused_by_index = {rep.rep_index: rep for rep in repetitions}
    frame_poses: dict[str, tuple[np.ndarray | None, ...]] = {}
    frame_sources: dict[str, tuple[str, ...]] = {}
    for view, rows in rows_by_view.items():
        poses: list[np.ndarray | None] = [_valid_pose(row) for row in rows]
        sources = ["single_view" if pose is not None else "unavailable" for pose in poses]
        segments = segments_by_view.get(view, [])
        for rep_index, segment in enumerate(segments):
            fused_rep = fused_by_index.get(rep_index)
            if fused_rep is None:
                continue
            start, end = segment.analysis_frames[0], segment.analysis_frames[-1]
            span = max(end - start, 1.0)
            for row_index, analysis_frame in zip(segment.row_indices, segment.analysis_frames):
                poses[int(row_index)] = _pose_at_progress(
                    fused_rep.coordinates, (float(analysis_frame) - start) / span
                )
                sources[int(row_index)] = "phase_fused"
        frame_poses[view] = tuple(poses)
        frame_sources[view] = tuple(sources)
    return MultiViewFusionResult(tuple(repetitions), frame_poses, frame_sources)


__all__ = [
    "FusedRepetition",
    "MultiViewFusionResult",
    "fuse_recorded_views",
]
