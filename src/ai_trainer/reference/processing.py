"""Geometry, repetition segmentation, and template aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import COMMON_JOINTS, FEATURE_NAMES


class ProcessingError(ValueError):
    """Raised when a sequence cannot safely become reference data."""


@dataclass(frozen=True)
class RepetitionWindow:
    start: int
    bottom: int
    end: int
    flexion_range_deg: float


@dataclass(frozen=True)
class ProcessedRepetition:
    positions: np.ndarray
    features: np.ndarray
    source_start: int
    source_bottom: int
    source_end: int
    bottom_phase: float
    flexion_range_deg: float
    valid_ratio: float


def _joint_index(name: str) -> int:
    return COMMON_JOINTS.index(name)


def _safe_unit(vector: np.ndarray, label: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-8:
        raise ProcessingError(f"Cannot determine {label}: degenerate vector")
    return vector / length


def _longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def interpolate_missing(
    points: np.ndarray, *, max_missing_gap_frames: int = 2
) -> tuple[np.ndarray, float]:
    """Interpolate internal/edge gaps independently for every coordinate."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ProcessingError("Expected skeleton coordinates with shape [frames, joints, 3]")
    if values.shape[0] < 2:
        raise ProcessingError("At least two frames are required")

    finite = np.isfinite(values)
    valid_ratio = float(finite.all(axis=2).mean())
    result = values.copy()
    frame_axis = np.arange(values.shape[0], dtype=np.float64)
    for joint in range(values.shape[1]):
        for axis in range(3):
            column = result[:, joint, axis]
            good = np.isfinite(column)
            if int(good.sum()) < 2:
                raise ProcessingError(
                    f"Joint {COMMON_JOINTS[joint]} axis {axis} has fewer than two valid values"
                )
            if not good.all():
                longest_gap = _longest_false_run(good)
                if longest_gap > max_missing_gap_frames:
                    raise ProcessingError(
                        f"Joint {COMMON_JOINTS[joint]} axis {axis} has a "
                        f"{longest_gap}-frame gap (maximum {max_missing_gap_frames})"
                    )
                column[~good] = np.interp(frame_axis[~good], frame_axis[good], column[good])
    return result, valid_ratio


def normalize_skeleton(
    points: np.ndarray, *, max_missing_gap_frames: int = 2
) -> tuple[np.ndarray, float]:
    """Hip-center, body-scale, and rigidly orient one skeleton sequence."""

    values, valid_ratio = interpolate_missing(
        points, max_missing_gap_frames=max_missing_gap_frames
    )
    left_hip = values[:, _joint_index("LHip")]
    right_hip = values[:, _joint_index("RHip")]
    neck = values[:, _joint_index("Neck")]
    hip_center = (left_hip + right_hip) / 2.0

    torso_lengths = np.linalg.norm(neck - hip_center, axis=1)
    scale = float(np.median(torso_lengths[np.isfinite(torso_lengths)]))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ProcessingError("Body scale is zero or invalid")

    centered = values - hip_center[:, None, :]
    edge_count = max(2, min(values.shape[0] // 5, 10))
    stance_indices = np.r_[0:edge_count, values.shape[0] - edge_count : values.shape[0]]

    x_vector = np.median(left_hip[stance_indices] - right_hip[stance_indices], axis=0)
    x_axis = _safe_unit(x_vector, "left-right axis")
    up_vector = np.median(neck[stance_indices] - hip_center[stance_indices], axis=0)
    up_vector = up_vector - np.dot(up_vector, x_axis) * x_axis
    y_axis = _safe_unit(up_vector, "vertical axis")
    z_axis = _safe_unit(np.cross(x_axis, y_axis), "front-back axis")
    # Recompute y so numerical error cannot leave a non-orthogonal basis.
    y_axis = _safe_unit(np.cross(z_axis, x_axis), "vertical axis")
    basis = np.column_stack((x_axis, y_axis, z_axis))

    normalized = np.einsum("tjc,ck->tjk", centered, basis) / scale
    # Hip is a virtual joint in the common schema and should be exactly at origin.
    normalized[:, _joint_index("Hip"), :] = 0.0
    return normalized.astype(np.float32), valid_ratio


def _angle(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> np.ndarray:
    first = a - vertex
    second = c - vertex
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.divide(
        np.einsum("ij,ij->i", first, second),
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float64),
        where=denominator > 1e-8,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def extract_features(points: np.ndarray) -> np.ndarray:
    """Extract per-frame biomechanical features from normalized coordinates."""

    p = np.asarray(points, dtype=np.float64)
    get = lambda name: p[:, _joint_index(name)]  # noqa: E731

    left_knee = _angle(get("LHip"), get("LKnee"), get("LAnkle"))
    right_knee = _angle(get("RHip"), get("RKnee"), get("RAnkle"))
    left_hip = _angle(get("LShoulder"), get("LHip"), get("LKnee"))
    right_hip = _angle(get("RShoulder"), get("RHip"), get("RKnee"))
    left_ankle = _angle(get("LKnee"), get("LAnkle"), get("LBigToe"))
    right_ankle = _angle(get("RKnee"), get("RAnkle"), get("RBigToe"))

    trunk = get("Neck") - get("Hip")
    trunk_lean = np.degrees(np.arctan2(trunk[:, 2], trunk[:, 1]))
    left_tracking = get("LKnee")[:, 0] - get("LBigToe")[:, 0]
    right_tracking = get("RKnee")[:, 0] - get("RBigToe")[:, 0]
    ankle_mid = (get("LAnkle") + get("RAnkle")) / 2.0
    pelvis_height = get("Hip")[:, 1] - ankle_mid[:, 1]
    stance_width = np.abs(get("LAnkle")[:, 0] - get("RAnkle")[:, 0])
    heel_asymmetry = np.abs(get("LHeel")[:, 1] - get("RHeel")[:, 1])

    features = np.column_stack(
        (
            left_knee,
            right_knee,
            left_hip,
            right_hip,
            left_ankle,
            right_ankle,
            trunk_lean,
            np.abs(left_knee - right_knee),
            np.abs(left_hip - right_hip),
            left_tracking,
            right_tracking,
            pelvis_height,
            stance_width,
            heel_asymmetry,
        )
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("Feature schema and implementation disagree")
    if not np.isfinite(features).all():
        raise ProcessingError("Derived biomechanical features contain invalid values")
    return features.astype(np.float32)


def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.astype(np.float64, copy=True)
    window = min(window, signal.size if signal.size % 2 else signal.size - 1)
    window = max(window, 1)
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def segment_repetitions(
    features: np.ndarray,
    *,
    min_flexion_deg: float = 25.0,
    min_frames: int = 15,
) -> list[RepetitionWindow]:
    """Split one annotated interval into complete squat repetitions."""

    if features.ndim != 2 or features.shape[1] < 2:
        raise ProcessingError("Expected a [frames, features] feature matrix")
    if features.shape[0] < min_frames:
        raise ProcessingError(
            f"Annotated interval has {features.shape[0]} frames; at least {min_frames} required"
        )

    knee_angle = np.mean(features[:, :2], axis=1)
    flexion = _smooth(180.0 - knee_angle, window=5)
    baseline = float(np.percentile(flexion, 15))
    maximum = float(np.max(flexion))
    excursion = maximum - baseline
    if excursion < min_flexion_deg:
        raise ProcessingError(
            f"Knee-flexion range {excursion:.1f}° is below the {min_flexion_deg:.1f}° minimum"
        )

    active_threshold = baseline + max(min_flexion_deg, 0.45 * excursion)
    active = flexion >= active_threshold
    groups: list[tuple[int, int]] = []
    group_start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and group_start is None:
            group_start = index
        elif not is_active and group_start is not None:
            groups.append((group_start, index - 1))
            group_start = None
    if group_start is not None:
        groups.append((group_start, len(active) - 1))

    peaks: list[int] = []
    for start, end in groups:
        active_values = flexion[start : end + 1]
        plateau = np.flatnonzero(active_values >= float(active_values.max()) - 0.25)
        peaks.append(start + int(round(float(np.mean(plateau)))))
    if not peaks:
        peaks = [int(np.argmax(flexion))]

    windows: list[RepetitionWindow] = []
    for peak_index, peak in enumerate(peaks):
        peak_height = float(flexion[peak])
        boundary_level = baseline + 0.20 * (peak_height - baseline)
        left_limit = 0 if peak_index == 0 else (peaks[peak_index - 1] + peak) // 2
        right_limit = len(flexion) - 1 if peak_index == len(peaks) - 1 else (peak + peaks[peak_index + 1]) // 2

        start = peak
        while start > left_limit and flexion[start] > boundary_level:
            start -= 1
        end = peak
        while end < right_limit and flexion[end] > boundary_level:
            end += 1

        if flexion[start] > boundary_level or flexion[end] > boundary_level:
            # The JSON interval contains only part of the repetition, or the
            # motion did not return to a standing/top state between peaks.
            continue
        if peak - start < 2 or end - peak < 2 or end - start + 1 < min_frames:
            continue
        local_baseline = float(max(flexion[start], flexion[end]))
        local_range = peak_height - local_baseline
        if local_range < min_flexion_deg:
            continue
        windows.append(
            RepetitionWindow(
                start=start,
                bottom=peak,
                end=end,
                flexion_range_deg=local_range,
            )
        )

    if not windows:
        raise ProcessingError("No complete air-squat repetition was detected")
    return windows


def resample(values: np.ndarray, target_frames: int) -> np.ndarray:
    """Linearly resample a time-first matrix/tensor."""

    if target_frames < 3:
        raise ProcessingError("target_frames must be at least 3")
    source = np.asarray(values, dtype=np.float64)
    if source.shape[0] < 2:
        raise ProcessingError("At least two source frames are required for resampling")
    old_axis = np.linspace(0.0, 1.0, source.shape[0])
    new_axis = np.linspace(0.0, 1.0, target_frames)
    flat = source.reshape(source.shape[0], -1)
    output = np.empty((target_frames, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(new_axis, old_axis, flat[:, column])
    return output.reshape((target_frames,) + source.shape[1:]).astype(np.float32)


def process_sequence(
    raw_points: np.ndarray,
    *,
    target_frames: int,
    min_flexion_deg: float,
    min_frames: int,
    min_valid_ratio: float,
    max_missing_gap_frames: int,
) -> list[ProcessedRepetition]:
    """Normalize, segment, and phase-resample an annotated sequence."""

    normalized, valid_ratio = normalize_skeleton(
        raw_points, max_missing_gap_frames=max_missing_gap_frames
    )
    if valid_ratio < min_valid_ratio:
        raise ProcessingError(
            f"Valid joint ratio {valid_ratio:.3f} is below the {min_valid_ratio:.3f} minimum"
        )
    features = extract_features(normalized)
    windows = segment_repetitions(
        features,
        min_flexion_deg=min_flexion_deg,
        min_frames=min_frames,
    )
    result: list[ProcessedRepetition] = []
    for window in windows:
        if target_frames % 2 == 0:
            raise ProcessingError("target_frames must be odd so the bottom phase has one center frame")
        half_frames = target_frames // 2 + 1
        descent_positions = resample(
            normalized[window.start : window.bottom + 1], half_frames
        )
        ascent_positions = resample(normalized[window.bottom : window.end + 1], half_frames)
        descent_features = resample(features[window.start : window.bottom + 1], half_frames)
        ascent_features = resample(features[window.bottom : window.end + 1], half_frames)
        phase_denominator = max(window.end - window.start, 1)
        result.append(
            ProcessedRepetition(
                positions=np.concatenate((descent_positions[:-1], ascent_positions), axis=0),
                features=np.concatenate((descent_features[:-1], ascent_features), axis=0),
                source_start=window.start,
                source_bottom=window.bottom,
                source_end=window.end,
                bottom_phase=(window.bottom - window.start) / phase_denominator,
                flexion_range_deg=window.flexion_range_deg,
                valid_ratio=valid_ratio,
            )
        )
    return result


def robust_keep_mask(feature_stack: np.ndarray, mad_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Flag gross trajectory outliers using distance to the median template."""

    count = feature_stack.shape[0]
    template = np.median(feature_stack, axis=0)
    scale = np.std(feature_stack.reshape(-1, feature_stack.shape[-1]), axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    distances = np.sqrt(np.mean(((feature_stack - template) / scale) ** 2, axis=(1, 2)))
    if count < 5 or mad_threshold <= 0:
        return np.ones(count, dtype=bool), distances.astype(np.float32)
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    if mad < 1e-9:
        return np.ones(count, dtype=bool), distances.astype(np.float32)
    robust_sigma = 1.4826 * mad
    return distances <= median + mad_threshold * robust_sigma, distances.astype(np.float32)


def aggregate_repetitions(
    repetitions: list[ProcessedRepetition],
    *,
    mad_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Create median/quantile templates and choose a real representative rep."""

    if not repetitions:
        raise ProcessingError("No repetitions are available for aggregation")
    positions = np.stack([item.positions for item in repetitions])
    features = np.stack([item.features for item in repetitions])
    keep, distances = robust_keep_mask(features, mad_threshold)
    if not keep.any():
        raise ProcessingError("All repetitions were rejected as outliers")

    kept_positions = positions[keep]
    kept_features = features[keep]
    kept_indices = np.flatnonzero(keep)
    representative_original_index = kept_indices[int(np.argmin(distances[keep]))]

    bottom_index = positions.shape[1] // 2
    phase = np.full(positions.shape[1], 2, dtype=np.int8)
    phase[:bottom_index] = 0
    phase[bottom_index] = 1

    arrays = {
        "positions_median": np.median(kept_positions, axis=0).astype(np.float32),
        "positions_q10": np.quantile(kept_positions, 0.10, axis=0).astype(np.float32),
        "positions_q90": np.quantile(kept_positions, 0.90, axis=0).astype(np.float32),
        "features_median": np.median(kept_features, axis=0).astype(np.float32),
        "features_q10": np.quantile(kept_features, 0.10, axis=0).astype(np.float32),
        "features_q90": np.quantile(kept_features, 0.90, axis=0).astype(np.float32),
        "representative_positions": positions[representative_original_index].astype(np.float32),
        "representative_features": features[representative_original_index].astype(np.float32),
        "phase": phase,
        "phase_progress": np.linspace(0.0, 1.0, positions.shape[1], dtype=np.float32),
        "bottom_index": np.asarray(bottom_index, dtype=np.int32),
        "representative_index": np.asarray(representative_original_index, dtype=np.int32),
    }
    return arrays, keep, distances
