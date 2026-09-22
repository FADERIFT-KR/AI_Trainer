"""AI Hub 8-camera layout identification utilities.

The dataset does not ship a convenient ``front/left/right`` label.  We infer it
from synchronized 2D keypoints and 3D ground truth instead of relying on camera
numbers alone:

* front/back candidates have the smallest Procrustes error to the raw 3D Y-Z
  projection;
* left/right candidates have the smallest error to the X-Z projection;
* face landmark ordering separates front from back;
* an orthographic 3D->2D fit and anatomical body axes separate the two sides.

The resulting mapping is deterministic for a fixed dataset, but the statistics
are written to the generated configuration so it remains auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ai_trainer.squat.aihub_zip import AiHubZip, JOINT_NAMES, SequenceKey
from ai_trainer.core.s1_capture.camera_id import evaluate_frame, sample_frame_indices

VIEW_FRONT = "front"
VIEW_LEFT = "left"
VIEW_RIGHT = "right"
VIEW_BACK = "back"
VIEWS = (VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT)
VIEW_LABEL_KO = {
    VIEW_FRONT: "정면",
    VIEW_LEFT: "좌측면",
    VIEW_RIGHT: "우측면",
    VIEW_BACK: "후면",
}

_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}


@dataclass(frozen=True)
class CameraViewMapping:
    front: int
    back: int
    left: int
    right: int
    statistics: dict[int, dict[str, float]]

    def as_dict(self) -> dict[str, int]:
        return {
            VIEW_FRONT: self.front,
            VIEW_BACK: self.back,
            VIEW_LEFT: self.left,
            VIEW_RIGHT: self.right,
        }


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def _body_axes(coords_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return anatomical (left, up, forward) unit axes for one raw 3D frame."""
    left = _unit(coords_3d[_IDX["LHip"]] - coords_3d[_IDX["RHip"]])
    if left is None:
        return None
    up_raw = coords_3d[_IDX["Neck"]] - coords_3d[_IDX["Hip"]]
    up = _unit(up_raw - np.dot(up_raw, left) * left)
    if up is None:
        return None

    # Nose has a stable forward component in this dataset.  Remove the already
    # identified lateral/vertical components before normalizing it.
    face = coords_3d[_IDX["Nose"]] - coords_3d[_IDX["Neck"]]
    face = face - np.dot(face, left) * left - np.dot(face, up) * up
    forward = _unit(face)
    if forward is None:
        forward = _unit(np.cross(left, up))
    if forward is None:
        return None
    return left, up, forward


def camera_position_components(
    coords_2d: np.ndarray, coords_3d: np.ndarray
) -> tuple[float, float, float] | None:
    """Estimate camera position in anatomical forward/left/up coordinates.

    A weak-perspective 3D->2D least-squares fit gives the image horizontal and
    vertical axes.  Their cross product points along the optical axis.  Image
    axes have a consistent convention across AI Hub cameras, so its negative is
    the subject-to-camera direction.  The caller may flip all directions once
    using the known front camera if a dataset exporter used the opposite image
    convention.
    """
    axes = _body_axes(coords_3d)
    if axes is None:
        return None
    left, up, forward = axes

    centered_3d = coords_3d - coords_3d.mean(axis=0, keepdims=True)
    centered_2d = coords_2d - coords_2d.mean(axis=0, keepdims=True)
    try:
        projection = np.linalg.lstsq(centered_3d, centered_2d, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    image_right = _unit(projection[:, 0])
    if image_right is None:
        return None
    image_down_raw = projection[:, 1] - np.dot(projection[:, 1], image_right) * image_right
    image_down = _unit(image_down_raw)
    if image_down is None:
        return None
    subject_to_camera = _unit(-np.cross(image_right, image_down))
    if subject_to_camera is None:
        return None
    return (
        float(np.dot(subject_to_camera, forward)),
        float(np.dot(subject_to_camera, left)),
        float(np.dot(subject_to_camera, up)),
    )


def face_plausibility(coords_2d: np.ndarray) -> bool:
    nose_x = coords_2d[_IDX["Nose"], 0]
    left_eye_x = coords_2d[_IDX["LEye"], 0]
    right_eye_x = coords_2d[_IDX["REye"], 0]
    return min(left_eye_x, right_eye_x) <= nose_x <= max(left_eye_x, right_eye_x)


def infer_camera_views(
    dataset: AiHubZip,
    sequences: Iterable[SequenceKey],
    frames_per_sequence: int = 5,
) -> CameraViewMapping:
    """Infer front/back/left/right camera ids from representative sequences."""
    accum: dict[int, dict[str, list[float]]] = {}
    for seq in sequences:
        frames_3d, coords_3d = dataset.read_3d(seq)
        for camera in dataset.list_cameras(seq):
            try:
                frames_2d, coords_2d = dataset.read_2d(seq, camera)
                annotation = dataset.read_annotation(seq, camera)
            except (FileNotFoundError, KeyError, ValueError):
                continue
            n_frames = min(len(frames_3d), len(frames_2d))
            if n_frames < 5:
                continue
            start, end = 0, n_frames - 1
            if annotation.get("annotations"):
                item = annotation["annotations"][0]
                start = max(0, min(int(item["start_frame"]), n_frames - 1))
                end = max(start, min(int(item["end_frame"]), n_frames - 1))
            stats = accum.setdefault(
                camera,
                {"XY": [], "XZ": [], "YZ": [], "face": [], "forward": [], "left": [], "up": []},
            )
            for frame in sample_frame_indices(start, end, frames_per_sequence):
                disparity = evaluate_frame(coords_2d[frame], coords_3d[frame])
                for projection in ("XY", "XZ", "YZ"):
                    value = disparity[projection]
                    if value is not None:
                        stats[projection].append(float(value))
                stats["face"].append(float(face_plausibility(coords_2d[frame])))
                direction = camera_position_components(coords_2d[frame], coords_3d[frame])
                if direction is not None:
                    stats["forward"].append(direction[0])
                    stats["left"].append(direction[1])
                    stats["up"].append(direction[2])

    if len(accum) < 4:
        raise ValueError("카메라 시점을 분류할 만큼 유효한 camera 데이터가 없습니다.")

    summary: dict[int, dict[str, float]] = {}
    for camera, values in accum.items():
        summary[camera] = {
            key: float(np.median(series)) if series else float("nan")
            for key, series in values.items()
        }
        summary[camera]["n_frames"] = float(len(values["YZ"]))

    front_back = sorted(summary, key=lambda camera: summary[camera]["YZ"])[:2]
    front = max(front_back, key=lambda camera: summary[camera]["face"])
    back = next(camera for camera in front_back if camera != front)

    # Resolve a possible global optical-axis sign convention using the camera
    # already identified as frontal.
    direction_sign = 1.0 if summary[front]["forward"] >= 0 else -1.0
    for camera in summary:
        summary[camera]["forward"] *= direction_sign
        summary[camera]["left"] *= direction_sign
        summary[camera]["up"] *= direction_sign

    side_candidates = sorted(
        (camera for camera in summary if camera not in front_back),
        key=lambda camera: summary[camera]["XZ"],
    )[:2]
    left = max(side_candidates, key=lambda camera: summary[camera]["left"])
    right = min(side_candidates, key=lambda camera: summary[camera]["left"])
    return CameraViewMapping(front=front, back=back, left=left, right=right, statistics=summary)


__all__ = [
    "CameraViewMapping",
    "VIEW_BACK",
    "VIEW_FRONT",
    "VIEW_LABEL_KO",
    "VIEW_LEFT",
    "VIEW_RIGHT",
    "VIEWS",
    "camera_position_components",
    "face_plausibility",
    "infer_camera_views",
]
