"""Post-session replay with deterministic error-region overlays.

The live pipeline stores the raw (undecorated) camera image and the matching
``common_2d`` landmarks.  This module joins those rows with the final per-rep
decision after all front/left/right measurements finish.  It deliberately does
not use a live score: only final erroneous repetitions receive a red overlay.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ai_trainer.squat.camera_views import VIEW_LABEL_KO, VIEWS
from ai_trainer.core.common_skeleton import (
    COMMON_BONE_COLORS_BGR,
    COMMON_BONE_INDEX_PAIRS,
    COMMON_JOINT_NAMES,
)
from ai_trainer.squat.multiview_fusion import fuse_recorded_views
from ai_trainer.core.render import TransformFn, draw_skeleton_panel

from ai_trainer.squat.game_ui.error_explain import ERROR_EXPLANATIONS


_IDX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
_NORMAL = "정상"


@dataclass(frozen=True)
class ReplaySpan:
    """One final erroneous repetition mapped to saved video-frame numbers."""

    start_video_frame: int
    end_video_frame: int
    label: str
    joints: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class ReplayView:
    view: str
    video_path: Path
    rows: tuple[dict, ...]
    fps: float
    spans: tuple[ReplaySpan, ...]
    fused_frames: tuple[np.ndarray | None, ...] = ()
    fusion_sources: tuple[str, ...] = ()


def error_joints(label: str, paper_violations: Iterable[dict] = ()) -> tuple[str, ...]:
    """Map final diagnostic wording to the anatomical region to draw in red."""
    explain = ERROR_EXPLANATIONS.get(label)
    joints: set[str] = set(explain.joints if explain is not None else ())
    text = label.casefold()
    if "발뒤꿈치" in text or "heel" in text:
        joints.update(("LHeel", "RHeel", "LAnkle", "RAnkle"))
    if "무릎" in text or "knee" in text:
        joints.update(("LKnee", "RKnee"))
    if "고관절" in text or "골반" in text or "허리" in text or "상체" in text or "hip" in text:
        joints.update(("Hip", "Neck", "LHip", "RHip"))
    if "발목" in text or "ankle" in text:
        joints.update(("LAnkle", "RAnkle", "LHeel", "RHeel"))

    # The cited-paper rules are side-view rules.  Their condition names remain
    # stable in the saved assessment even if message text is changed later.
    for violation in paper_violations:
        condition = str(violation.get("condition", ""))
        if "1" in condition:  # knee-hip-shoulder angle
            joints.update(("Hip", "LHip", "RHip", "Neck", "LShoulder", "RShoulder", "LKnee", "RKnee"))
        elif "2" in condition:  # horizontal thigh
            joints.update(("LHip", "RHip", "LKnee", "RKnee"))
        elif "3" in condition:  # knee-toe alignment
            joints.update(("LKnee", "RKnee", "LBigToe", "RBigToe"))
    # A non-normal result without a typed error must not imply a precise body
    # part.  Highlight the pelvis/leg chain rather than inventing a diagnosis.
    if not joints:
        joints.update(("Hip", "LHip", "RHip", "LKnee", "RKnee"))
    return tuple(name for name in COMMON_JOINT_NAMES if name in joints)


def _is_error(rep: dict) -> bool:
    return (
        rep.get("sequence_class") not in (None, _NORMAL)
        or bool(rep.get("paper_posture_messages"))
        or rep.get("display_class") not in (None, _NORMAL)
    )


def _rows_for_analysis_range(rows: list[dict], start: int, end: int) -> list[int]:
    return [
        int(row["video_frame"])
        for row in rows
        if isinstance(row.get("analysis_frame"), int) and start <= row["analysis_frame"] <= end
    ]


def build_replay_views(recording_directory: str | Path, repetitions: Iterable[dict]) -> list[ReplayView]:
    """Create replay plans from the temporary recording and final UI decisions."""
    directory = Path(recording_directory)
    final_reps = list(repetitions)
    loaded: dict[str, tuple[dict, list[dict]]] = {}
    for view in VIEWS:
        metadata_path = directory / f"{view}.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (directory / metadata["timeline_file"])
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            loaded[view] = (metadata, rows)
    fusion = fuse_recorded_views(
        {view: item[1] for view, item in loaded.items()},
        {view: float(item[0]["video_fps"]) for view, item in loaded.items()},
    )
    replay_views: list[ReplayView] = []
    for view in VIEWS:
        if view not in loaded:
            continue
        metadata, rows = loaded[view]
        events = [row for row in rows if row.get("completed_rep") is not None]
        reps = sorted((rep for rep in final_reps if rep.get("view_mode") == view),
                      key=lambda rep: int(rep.get("view_rep_index", 0)))
        spans: list[ReplaySpan] = []
        for event, rep in zip(events, reps):
            if not _is_error(rep):
                continue
            frame_range = event["completed_rep"].get("frame_range")
            if not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2:
                continue
            frame_numbers = _rows_for_analysis_range(rows, int(frame_range[0]), int(frame_range[1]))
            if not frame_numbers:
                # Old recordings may lack analysis_frame on the initial rows.
                # The event row still identifies the end; make a conservative
                # one-second window rather than applying an overlay elsewhere.
                end_frame = int(event["video_frame"])
                start_frame = max(0, end_frame - max(1, int(float(metadata["video_fps"]))))
            else:
                start_frame, end_frame = min(frame_numbers), max(frame_numbers)
            paper = rep.get("condition_assessment", {}).get("paper_posture_violations", [])
            label = str(rep.get("display_class") or rep.get("sequence_class") or "자세 오류")
            message = str(rep.get("fail_reason") or "최종 판정에서 오류로 확인된 부위")
            spans.append(ReplaySpan(start_frame, end_frame, label, error_joints(label, paper), message))
        if spans:
            replay_views.append(ReplayView(
                view=view,
                video_path=directory / metadata["video_file"],
                rows=tuple(rows),
                fps=float(metadata["video_fps"]),
                spans=tuple(spans),
                fused_frames=fusion.frame_poses.get(view, ()),
                fusion_sources=fusion.frame_sources.get(view, ()),
            ))
    return replay_views


def annotate_replay_frame(frame_bgr: np.ndarray, row: dict, spans: Iterable[ReplaySpan]) -> np.ndarray:
    """Return a frame with red outlines only if it belongs to a final error span."""
    result = frame_bgr.copy()
    frame_number = int(row.get("video_frame", -1))
    active = [span for span in spans if span.start_video_frame <= frame_number <= span.end_video_frame]
    if not active:
        return result
    points = np.asarray(row.get("common_2d"), dtype=float)
    if points.shape != (len(COMMON_JOINT_NAMES), 2) or not np.isfinite(points).all():
        return result
    joints = {name for span in active for name in span.joints}
    indexes = {_IDX[name] for name in joints}
    for first, second in COMMON_BONE_INDEX_PAIRS:
        if first in indexes and second in indexes:
            p1, p2 = tuple(np.rint(points[first]).astype(int)), tuple(np.rint(points[second]).astype(int))
            cv2.line(result, p1, p2, (0, 0, 255), 5, cv2.LINE_AA)
    for index in indexes:
        center = tuple(np.rint(points[index]).astype(int))
        cv2.circle(result, center, 13, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(result, center, 13, (255, 255, 255), 2, cv2.LINE_AA)
    label = active[0].label
    message = active[0].message
    cv2.rectangle(result, (8, 8), (min(result.shape[1] - 8, 720), 74), (0, 0, 210), -1, cv2.LINE_AA)
    cv2.putText(result, f"ERROR: {label}", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, message[:72], (18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def active_error_joints(row: dict, spans: Iterable[ReplaySpan]) -> tuple[str, ...]:
    """Return final-error joints active at this saved video frame."""
    frame_number = int(row.get("video_frame", -1))
    joints = {
        name
        for span in spans
        if span.start_video_frame <= frame_number <= span.end_video_frame
        for name in span.joints
    }
    return tuple(name for name in COMMON_JOINT_NAMES if name in joints)


def project_fused_pose(coordinates: np.ndarray) -> np.ndarray:
    """Project canonical xyz into a stable oblique view where depth remains visible."""
    pose = np.asarray(coordinates, dtype=float)
    if pose.shape != (len(COMMON_JOINT_NAMES), 3):
        raise ValueError("coordinates must have shape (18,3)")
    yaw = np.deg2rad(32.0)
    horizontal = pose[:, 0] * np.cos(yaw) + pose[:, 2] * np.sin(yaw)
    vertical = pose[:, 1] - 0.18 * pose[:, 2]
    return np.column_stack((horizontal, vertical))


def render_fused_skeleton_frame(
    coordinates: np.ndarray | None,
    transform: TransformFn | None,
    active_joints: Iterable[str] = (),
    source: str = "phase_fused",
    width: int = 560,
    height: int = 560,
) -> np.ndarray:
    """Render the final canonical skeleton and the same red error regions."""
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    if coordinates is None or transform is None:
        cv2.putText(canvas, "3D SKELETON UNAVAILABLE", (45, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 160, 175), 2, cv2.LINE_AA)
        return canvas
    projected = project_fused_pose(coordinates)
    points_px = transform(projected)
    active_indexes = {_IDX[name] for name in active_joints if name in _IDX}
    colors = [
        (0, 0, 255) if first in active_indexes or second in active_indexes else color
        for (first, second), color in zip(COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR)
    ]
    footer = ("FRONT + LEFT + RIGHT / PHASE ALIGNED"
              if source == "phase_fused" else "CURRENT VIEW ONLY / UNMATCHED PHASE")
    draw_skeleton_panel(
        canvas, (0, 0), width, height, points_px,
        "FINAL XYZ SKELETON (OBLIQUE VIEW)", footer,
        COMMON_BONE_INDEX_PAIRS, colors,
    )
    for index in active_indexes:
        if not np.isfinite(points_px[index]).all():
            continue
        center = tuple(np.rint(points_px[index]).astype(int))
        cv2.circle(canvas, center, 10, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 10, (255, 255, 255), 2, cv2.LINE_AA)
    # Canonical body axes make it explicit that this is xyz, not a second 2-D
    # landmark overlay.  OpenCV's built-in font is ASCII-only.
    cv2.putText(canvas, "x: lateral   y: vertical   z: depth", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 210), 1, cv2.LINE_AA)
    return canvas


def view_title(view: str) -> str:
    return f"{VIEW_LABEL_KO[view]} 오류 동작 재생"


__all__ = [
    "ReplaySpan", "ReplayView", "active_error_joints", "annotate_replay_frame",
    "build_replay_views", "error_joints", "project_fused_pose",
    "render_fused_skeleton_frame", "view_title",
]
