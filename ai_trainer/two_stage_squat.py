"""Normal-template DTW gate and path-aligned inputs for squat diagnosis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .camera_views import VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT, VIEWS
from .common_skeleton import COMMON_JOINT_NAMES
from .normalization import hip_center_3d, leg_length_scale

T_REF = 64
N_JOINTS = len(COMMON_JOINT_NAMES)
ERROR_CLASSES = ("발뒤꿈치오류", "엉덩이하방오류", "고관절오류")
PARTS = ("발목/발", "무릎", "고관절", "몸통")
# Dataset annotations are error types, not independently annotated body parts.
# These are explicitly weak supervision targets for the second head.
PART_TARGETS = {
    "발뒤꿈치오류": (1, 0, 0, 0),
    "엉덩이하방오류": (0, 1, 1, 0),
    "고관절오류": (0, 0, 1, 1),
}
# Squat form is dominated by legs, feet and trunk. Variable arm placement
# otherwise overwhelms the normal-template distance despite correct legs.
_JOINT_WEIGHTS = {
    "LShoulder": 0.25, "RShoulder": 0.25,
    "LElbow": 0.0, "RElbow": 0.0, "LWrist": 0.0, "RWrist": 0.0,
    "LHip": 1.0, "RHip": 1.0,
    "LKnee": 1.5, "RKnee": 1.5,
    "LAnkle": 1.5, "RAnkle": 1.5,
    "LHeel": 1.0, "RHeel": 1.0,
    "LBigToe": 1.0, "RBigToe": 1.0,
    "Hip": 0.0, "Neck": 0.75,
}
JOINT_WEIGHTS = np.array([_JOINT_WEIGHTS[name] for name in COMMON_JOINT_NAMES], dtype=np.float32)
_INDEX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}


def visible_joint_weights(view: str) -> np.ndarray:
    """Use the camera-near leg and shoulder for side-view squat comparison."""
    if view not in VIEWS:
        raise ValueError(f"Unsupported camera view: {view}")
    weights = JOINT_WEIGHTS.copy()
    if view != VIEW_FRONT:
        far = "R" if view == VIEW_LEFT else "L"
        for name in ("Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle", "Heel", "BigToe"):
            weights[_INDEX[f"{far}{name}"]] = 0.0
        # These midpoint landmarks also depend on the far side.
        weights[_INDEX["Hip"]] = weights[_INDEX["Neck"]] = 0.0
    return weights


def mask_occluded_joints(track: np.ndarray, view: str) -> np.ndarray:
    """Zero side-view joints not used by the separately trained graph model."""
    if view not in VIEWS:
        raise ValueError(f"Unsupported camera view: {view}")
    result = np.asarray(track, dtype=np.float32).copy()
    if result.shape[-2:] != (N_JOINTS, 3):
        raise ValueError("Expected (...,18,3) skeleton")
    if view != VIEW_FRONT:
        result[..., visible_joint_weights(view) == 0, :] = 0.0
    return result


def normalize_track(coords: np.ndarray, view: str = VIEW_FRONT) -> np.ndarray:
    """Center each frame at the pelvis and set median bilateral leg length to 1."""
    x = np.asarray(coords, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (N_JOINTS, 3) or len(x) < 2:
        raise ValueError("Expected (T,18,3) with at least two frames")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite skeleton coordinates")
    if view not in VIEWS:
        raise ValueError(f"Unsupported camera view: {view}")
    if view == VIEW_FRONT:
        centered, _ = hip_center_3d(x)
        lengths = leg_length_scale(centered)
    else:
        near = "L" if view == VIEW_LEFT else "R"
        hip, knee, ankle = (_INDEX[f"{near}{name}"] for name in ("Hip", "Knee", "Ankle"))
        centered = x - x[:, hip:hip + 1]
        lengths = (np.linalg.norm(centered[:, hip] - centered[:, knee], axis=-1)
                   + np.linalg.norm(centered[:, knee] - centered[:, ankle], axis=-1))
    scale = float(np.median(lengths))
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("Degenerate leg length")
    return (centered / scale).astype(np.float32)


def resample_track(coords: np.ndarray, length: int = T_REF) -> np.ndarray:
    x = np.asarray(coords, dtype=np.float32)
    if len(x) < 2 or length < 2:
        raise ValueError("Need at least two input and output frames")
    old_t = np.linspace(0.0, 1.0, len(x))
    new_t = np.linspace(0.0, 1.0, length)
    flat = x.reshape(len(x), -1)
    return np.stack([np.interp(new_t, old_t, flat[:, k]) for k in range(flat.shape[1])], axis=1).reshape(length, N_JOINTS, 3).astype(np.float32)


def _cost_matrix(user: np.ndarray, reference: np.ndarray, view: str = VIEW_FRONT) -> np.ndarray:
    # Weighted pose distance in leg-length units; arm position is not squat form.
    weights = visible_joint_weights(view)
    scale = np.sqrt(weights)[None, :, None]
    a = (user * scale).reshape(len(user), -1)
    b = (reference * scale).reshape(len(reference), -1)
    return cdist(a, b, metric="euclidean") / np.sqrt(float(weights.sum()))


def dtw_path(user: np.ndarray, reference: np.ndarray,
             view: str = VIEW_FRONT) -> tuple[float, np.ndarray]:
    """Return mean path cost and (user_index, reference_index) optimal path."""
    a, b = normalize_track(user, view), normalize_track(reference, view)
    cost = _cost_matrix(a, b, view)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    predecessor = np.zeros((n, m), dtype=np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
            step = int(np.argmin(options))
            acc[i, j] = cost[i - 1, j - 1] + options[step]
            predecessor[i - 1, j - 1] = step
    i, j = n - 1, m - 1
    path = []
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        step = predecessor[i, j]
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(acc[n, m] / len(path)), np.asarray(path, dtype=np.int32)


def warp_to_reference(user: np.ndarray, path: np.ndarray, length: int = T_REF,
                      view: str = VIEW_FRONT) -> np.ndarray:
    """Average all user frames assigned to each reference frame by the DTW path."""
    x = normalize_track(user, view)
    sums = np.zeros((length, N_JOINTS, 3), dtype=np.float64)
    counts = np.zeros(length, dtype=np.int32)
    if len(path) == 0 or path[:, 1].min() != 0 or path[:, 1].max() != length - 1:
        raise ValueError("Path does not cover the reference")
    for source, target in path:
        sums[target] += x[source]
        counts[target] += 1
    if np.any(counts == 0):
        raise ValueError("Incomplete DTW path")
    return normalize_track((sums / counts[:, None, None]).astype(np.float32), view)


def build_normal_template(normal_tracks: list[np.ndarray], length: int = T_REF, iterations: int = 3) -> np.ndarray:
    """DTW barycenter: align multiple correct repetitions, then average framewise."""
    if len(normal_tracks) < 2:
        raise ValueError("At least two correct repetitions are required")
    tracks = [normalize_track(track) for track in normal_tracks]
    # A medoid resampled to T_ref avoids starting from a blurred arithmetic mean.
    seeds = [resample_track(x, length) for x in tracks]
    # Limit medoid candidates while retaining every track in the barycenter.
    candidate_indices = np.unique(np.linspace(0, len(seeds) - 1, min(8, len(seeds)), dtype=int))
    distance_sums = [sum(dtw_path(seeds[i], other)[0] for other in seeds) for i in candidate_indices]
    template = seeds[int(candidate_indices[int(np.argmin(distance_sums))])]
    for _ in range(iterations):
        aligned = [warp_to_reference(track, dtw_path(track, template)[1], length) for track in tracks]
        template = normalize_track(np.mean(aligned, axis=0))
    return template


@dataclass(frozen=True)
class GateResult:
    distance: float
    threshold: float
    passed: bool
    aligned: np.ndarray


@dataclass
class NormalTemplateGate:
    track: np.ndarray
    threshold: float
    view: str = VIEW_FRONT

    def __post_init__(self):
        self.track = normalize_track(self.track, self.view)
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("A positive calibrated threshold is required")

    def assess(self, user: np.ndarray) -> GateResult:
        distance, path = dtw_path(user, self.track, self.view)
        return GateResult(distance, self.threshold, distance <= self.threshold,
                          warp_to_reference(user, path, len(self.track), self.view))

    @classmethod
    def load(cls, path: str | Path) -> "NormalTemplateGate":
        with np.load(path) as data:
            return cls(data["track"], float(data["threshold"]))

    @classmethod
    def load_side(cls, path: str | Path, view: str) -> "NormalTemplateGate":
        if view not in (VIEW_LEFT, VIEW_RIGHT):
            raise ValueError("A side camera view is required")
        with np.load(path) as data:
            return cls(data["track"], float(data[f"threshold_{view}"]), view)


def choose_threshold(distances: np.ndarray, is_normal: np.ndarray) -> tuple[float, dict]:
    """Maximize calibration-set binary F1; ties favor lower false-pass rate."""
    values = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(is_normal, dtype=bool)
    if len(values) != len(labels) or not labels.any() or labels.all() or not np.isfinite(values).all():
        raise ValueError("Both normal and error calibration examples are required")
    candidates = np.unique(values)
    best = None
    for threshold in candidates:
        passing = values <= threshold
        tp = int(np.sum(passing & labels))
        fp = int(np.sum(passing & ~labels))
        fn = int(np.sum(~passing & labels))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        rank = (f1, -fp, -threshold)
        if best is None or rank > best[0]:
            best = (rank, float(threshold), {"f1": f1, "normal_recall": tp / labels.sum(),
                                            "error_false_pass_rate": fp / (~labels).sum(),
                                            "n_normal": int(labels.sum()), "n_error": int((~labels).sum())})
    return best[1], best[2]
