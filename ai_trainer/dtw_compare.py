"""Phase-aware Weighted DTW 비교 엔진.

cost(t,r) = sum_k w_k * d_k(feature_k(t), feature_k(r))

phase(준비/하강/최저점/상승/종료)별로 독립적으로 DTW를 계산한 뒤
D_total = sum_p phase_weight[p] * D_phase[p] 로 합산한다.
시퀀스 길이가 phase마다 다르므로, phase별 DTW distance는 정렬 경로 길이로
나눠 정규화해 짧은/긴 phase 사이의 스케일을 맞춘다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .features import FEATURE_NAMES

PHASES = ["준비", "하강", "최저점", "상승", "종료"]


# UI에는 표시하지만 자세 유사도 판정에는 사용하지 않는 feature.
EXCLUDED_COMPARISON_FEATURES = frozenset({
    "ankle_angle", "joint_velocity", "pelvis_trajectory",
})

DTW_ALGORITHMS = ("dtw", "ddtw")


def _validate_algorithm(algorithm: str) -> str:
    """Validate and normalize the alignment algorithm name."""
    normalized = str(algorithm).strip().lower()
    if normalized not in DTW_ALGORITHMS:
        raise ValueError(
            f"Unsupported DTW algorithm {algorithm!r}; "
            f"choose one of {', '.join(DTW_ALGORITHMS)}"
        )
    return normalized


def derivative_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return temporal derivatives with the same shape as each feature.

    Centered differences are used for interior frames and one-sided
    differences at the endpoints (``numpy.gradient``).  A one-frame sequence
    has no direction and therefore receives an all-zero derivative.  Non-finite
    values are made finite so SciPy's distance functions remain well-defined.
    """
    result: dict[str, np.ndarray] = {}
    for name, values in features.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            raise ValueError(f"Feature {name!r} must have a time axis")
        if array.shape[0] <= 1:
            derivative = np.zeros_like(array, dtype=np.float64)
        else:
            derivative = np.gradient(array, axis=0)
        result[name] = np.nan_to_num(
            derivative, nan=0.0, posinf=0.0, neginf=0.0
        )
    return result


def load_dtw_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_weights(config: dict, weight_profile: str, class_label: str | None) -> dict[str, float]:
    """weight_profile: 'D_full_weighted' | 'E_full_uniform' | 'A_coords_only' | 'B_angles_only' | 'C_coords_plus_angles'."""
    if weight_profile == "D_full_weighted":
        w = dict(config["base_weights"])
        if class_label and class_label in config["class_overrides"]:
            w.update(config["class_overrides"][class_label])
        return w
    if weight_profile == "E_full_uniform":
        return dict(config["uniform_weights"])
    ablation = config["ablation_configs"].get(weight_profile)
    if not isinstance(ablation, dict):
        raise ValueError(f"알 수 없는 weight_profile: {weight_profile}")
    return {name: ablation.get(name, 0.0) for name in FEATURE_NAMES}


def _dtw_dp(
    cost: np.ndarray, warping_window_ratio: float | None = None
) -> tuple[float, int]:
    """cost: (n,m) 프레임쌍 비용행렬 -> (누적 최소비용, 정렬 경로 길이 근사(n+m))."""
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    radius = None
    if warping_window_ratio is not None:
        if not 0.0 <= float(warping_window_ratio) <= 1.0:
            raise ValueError("warping_window_ratio must be between 0 and 1")
        radius = max(1, int(np.ceil(max(n, m) * float(warping_window_ratio))))
    for i in range(1, n + 1):
        row_cost = cost[i - 1]
        if radius is None:
            j_start, j_end = 1, m
        else:
            center = 1.0 if n <= 1 else 1.0 + (i - 1) * (m - 1) / (n - 1)
            j_start = max(1, int(np.floor(center - radius)))
            j_end = min(m, int(np.ceil(center + radius)))
        for j in range(j_start, j_end + 1):
            d[i, j] = row_cost[j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m]), n + m


def _resample_phase_features(
    features: dict[str, np.ndarray], target_frames: int
) -> dict[str, np.ndarray]:
    """Linearly resample every feature onto one shared normalized time axis."""
    if target_frames < 2:
        raise ValueError("phase_resample_frames values must be at least 2")
    source_frames = next(iter(features.values())).shape[0]
    if source_frames == target_frames:
        return features
    if source_frames <= 0:
        return features
    if source_frames == 1:
        return {
            name: np.repeat(np.asarray(values), target_frames, axis=0)
            for name, values in features.items()
        }
    source_x = np.linspace(0.0, 1.0, source_frames)
    target_x = np.linspace(0.0, 1.0, target_frames)
    result: dict[str, np.ndarray] = {}
    for name, values in features.items():
        array = np.asarray(values, dtype=np.float64)
        flat = array.reshape(source_frames, -1)
        resampled = np.stack(
            [np.interp(target_x, source_x, flat[:, column]) for column in range(flat.shape[1])],
            axis=1,
        )
        result[name] = resampled.reshape((target_frames,) + array.shape[1:])
    return result


def phase_slice(features: dict[str, np.ndarray], bounds: dict, phase: str) -> dict[str, np.ndarray]:
    s, e = bounds[phase]
    return {name: arr[s:e] for name, arr in features.items()}


def weighted_frame_cost_matrix(
    feat_a: dict[str, np.ndarray], feat_b: dict[str, np.ndarray], weights: dict[str, float], config: dict
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """반환: (합산 cost 행렬, {feature_name: weight*거리행렬} — 기여도 분석용)."""
    n = next(iter(feat_a.values())).shape[0]
    m = next(iter(feat_b.values())).shape[0]
    total = np.zeros((n, m))
    per_feature: dict[str, np.ndarray] = {}
    for name in FEATURE_NAMES:
        if name in EXCLUDED_COMPARISON_FEATURES:
            continue
        w = weights.get(name, 0.0)
        if w == 0.0 or n == 0 or m == 0:
            continue
        metric = config["features"][name]["metric"]
        d = cdist(feat_a[name], feat_b[name], metric=metric)
        d = np.nan_to_num(d, nan=0.0)  # cosine dist가 영벡터일 때 nan 방지
        contrib = w * d
        total += contrib
        per_feature[name] = contrib
    return total, per_feature


def phase_aware_weighted_dtw(
    feat_a: dict[str, np.ndarray],
    bounds_a: dict,
    feat_b: dict[str, np.ndarray],
    bounds_b: dict,
    weights: dict[str, float],
    config: dict,
    algorithm: str = "dtw",
    class_label: str | None = None,
) -> dict:
    """두 시퀀스 사이 phase별 DTW distance와 총합, feature별 기여도를 계산한다."""
    algorithm = _validate_algorithm(algorithm)
    # Compute derivatives before phase slicing so a phase boundary does not
    # create an artificial slope solely because the neighbouring frame is in
    # another phase.
    prepared_a = feat_a if algorithm == "dtw" else derivative_features(feat_a)
    prepared_b = feat_b if algorithm == "dtw" else derivative_features(feat_b)
    per_phase_dist: dict[str, float | None] = {}
    per_feature_contrib: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    total = 0.0
    phase_weights = config.get("class_phase_weights", {}).get(
        class_label, config["phase_weights"]
    )
    alignment = config.get("dtw_alignment", {})
    resample_frames = alignment.get("phase_resample_frames", {})
    max_length_ratio = float(alignment.get("max_phase_length_ratio", np.inf))
    warping_window_ratio = alignment.get("warping_window_ratio")
    length_ratio_by_phase: dict[str, float | None] = {}
    invalid_phases: list[str] = []

    for phase in PHASES:
        pa = phase_slice(prepared_a, bounds_a, phase)
        pb = phase_slice(prepared_b, bounds_b, phase)
        n = next(iter(pa.values())).shape[0]
        m = next(iter(pb.values())).shape[0]
        if n == 0 or m == 0:
            per_phase_dist[phase] = None
            length_ratio_by_phase[phase] = None
            continue
        length_ratio = max(n, m) / max(1, min(n, m))
        length_ratio_by_phase[phase] = float(length_ratio)
        if length_ratio > max_length_ratio:
            per_phase_dist[phase] = float("inf")
            invalid_phases.append(phase)
            continue
        target_frames = resample_frames.get(phase)
        if target_frames is not None:
            pa = _resample_phase_features(pa, int(target_frames))
            pb = _resample_phase_features(pb, int(target_frames))
        cost, per_feature = weighted_frame_cost_matrix(pa, pb, weights, config)
        d, path_len = _dtw_dp(cost, warping_window_ratio)
        d_norm = d / path_len
        per_phase_dist[phase] = d_norm
        total += phase_weights.get(phase, 1.0) * d_norm

        # feature 기여도: 해당 phase의 DTW 최적 경로 전체를 다시 추적하기보다,
        # 비용행렬 평균으로 근사(연산량을 줄이면서도 상대적 크기 비교에는 충분).
        for name, contrib in per_feature.items():
            per_feature_contrib[name] += phase_weights.get(phase, 1.0) * float(contrib.mean())

    if invalid_phases:
        total = float("inf")
    return {
        "total": total,
        "per_phase": per_phase_dist,
        "per_feature_contrib": per_feature_contrib,
        "length_ratio_by_phase": length_ratio_by_phase,
        "invalid_phases": invalid_phases,
        "alignment_valid": not invalid_phases and np.isfinite(total),
    }


def multi_reference_distance(
    query_feat: dict[str, np.ndarray],
    query_bounds: dict,
    medoids: list[dict],  # [{feat, bounds, meta}]
    weights: dict[str, float],
    config: dict,
    top_k: int = 2,
    algorithm: str = "dtw",
    class_label: str | None = None,
) -> dict:
    """query 하나를 한 클래스의 medoid 전체와 비교, min/top-k 평균 두 방식을 모두 반환."""
    algorithm = _validate_algorithm(algorithm)
    results = []
    for med in medoids:
        r = phase_aware_weighted_dtw(
            query_feat,
            query_bounds,
            med["feat"],
            med["bounds"],
            weights,
            config,
            algorithm=algorithm,
            class_label=class_label,
        )
        results.append({"medoid": med["meta"], "dtw": r})

    dists = np.array([r["dtw"]["total"] for r in results])
    order = np.argsort(dists)
    best = results[order[0]]
    top_k_actual = min(top_k, len(dists))
    mean_topk = float(dists[order[:top_k_actual]].mean())

    return {
        "algorithm": algorithm,
        "min_distance": float(dists[order[0]]),
        "mean_topk_distance": mean_topk,
        "best_medoid": best["medoid"],
        "best_detail": best["dtw"],
        "all": results,
    }
