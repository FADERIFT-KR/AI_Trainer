"""DTW distance -> 0~100 자세 점수 변환.

임의 고정 선형식 대신, 정상 reference distance 분포(하위 percentile)와
오류 시퀀스 distance 분포(중앙값)를 기준으로 두 앵커를 잡는다.
"""
from __future__ import annotations

import numpy as np


def _balanced_threshold(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    positive_when_high: bool,
) -> tuple[float, float]:
    """Return the threshold that maximizes binary balanced accuracy."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]
    if values.size == 0 or not labels.any() or labels.all():
        raise ValueError("threshold 보정에는 양성/음성 표본이 모두 필요합니다.")

    unique = np.unique(values)
    candidates = np.concatenate(
        [
            unique[:1] - 1e-9,
            (unique[:-1] + unique[1:]) / 2.0,
            unique[-1:] + 1e-9,
        ]
    )
    best_threshold = float(candidates[0])
    best_balanced_accuracy = -1.0
    for threshold in candidates:
        predicted = values >= threshold if positive_when_high else values <= threshold
        sensitivity = float(np.mean(predicted[labels]))
        specificity = float(np.mean(~predicted[~labels]))
        balanced_accuracy = (sensitivity + specificity) * 0.5
        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_threshold = float(threshold)
    return best_threshold, best_balanced_accuracy


def fit_score_calibration(normal_distances: np.ndarray, error_distances: np.ndarray) -> dict:
    normal_distances = np.asarray(normal_distances, dtype=np.float64)
    error_distances = np.asarray(error_distances, dtype=np.float64)
    normal_distances = normal_distances[np.isfinite(normal_distances)]
    error_distances = error_distances[np.isfinite(error_distances)]
    if normal_distances.size == 0 or error_distances.size == 0:
        raise ValueError("점수 보정에는 정상/오류 distance가 각각 1개 이상 필요합니다.")

    lo = float(np.percentile(normal_distances, 10))  # 상위(좋은) 앵커: 정상 시퀀스 중에서도 잘한 축에 속함
    hi = float(np.percentile(error_distances, 50))  # 하위(나쁜) 앵커: 오류 시퀀스의 전형적인 거리
    if hi <= lo:
        hi = lo + 1e-6

    # 정상 판정 threshold는 임의 상수가 아니라 calibration 표본에서 정상 재현율과
    # 오류 특이도의 평균(balanced accuracy)을 최대화하는 distance로 정한다.
    distances = np.concatenate([normal_distances, error_distances])
    labels = np.concatenate(
        [
            np.ones(normal_distances.size, dtype=bool),
            np.zeros(error_distances.size, dtype=bool),
        ]
    )
    best_threshold, best_balanced_accuracy = _balanced_threshold(
        distances,
        labels,
        positive_when_high=False,
    )

    calibration = {
        "lo": lo,
        "hi": hi,
        "normal_distance_threshold": best_threshold,
        "calibration_balanced_accuracy": best_balanced_accuracy,
    }
    calibration["normal_match_threshold"] = distance_to_score(best_threshold, calibration)
    return calibration


def distance_to_score(d: float, calib: dict) -> float:
    score = 100.0 * (calib["hi"] - d) / (calib["hi"] - calib["lo"])
    return float(np.clip(score, 0.0, 100.0))


def fit_binary_logistic_calibration(
    features: np.ndarray,
    normal_labels: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    *,
    l2: float = 0.5,
    max_iter: int = 100,
) -> dict:
    """Fit a small regularized classifier for the normal/error first stage.

    The calibration is deliberately NumPy-only so the runtime does not gain a
    scikit-learn dependency. Non-finite DTW values are replaced by robust,
    per-column caps learned here and stored for identical live inference.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(normal_labels, dtype=np.float64).reshape(-1)
    names = tuple(str(name) for name in feature_names)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[1] != len(names):
        raise ValueError("binary calibration feature shape가 올바르지 않습니다.")
    if x.shape[0] < 2 or np.all(y == y[0]):
        raise ValueError("binary calibration에는 정상/오류 표본이 모두 필요합니다.")

    impute = np.zeros(x.shape[1], dtype=np.float64)
    filled = x.copy()
    for column in range(x.shape[1]):
        finite = filled[np.isfinite(filled[:, column]), column]
        if finite.size == 0:
            cap = 0.0
        else:
            q25, median, q75 = np.percentile(finite, [25.0, 50.0, 75.0])
            cap = float(median + 3.0 * max(q75 - q25, 1e-6))
        impute[column] = cap
        filled[~np.isfinite(filled[:, column]), column] = cap

    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (filled - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])

    params = np.zeros(design.shape[1], dtype=np.float64)
    prevalence = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    params[0] = float(np.log(prevalence / (1.0 - prevalence)))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    for _ in range(max(1, int(max_iter))):
        logits = np.clip(design @ params, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        weights = np.maximum(probability * (1.0 - probability), 1e-6)
        gradient = design.T @ (probability - y) + penalty @ params
        hessian = design.T @ (weights[:, None] * design) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        params -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break

    probability = 1.0 / (
        1.0 + np.exp(-np.clip(design @ params, -40.0, 40.0))
    )
    threshold, balanced_accuracy = _balanced_threshold(
        probability,
        y.astype(bool),
        positive_when_high=True,
    )
    return {
        "feature_names": list(names),
        "impute": impute.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(params[0]),
        "coefficients": params[1:].tolist(),
        "probability_threshold": threshold,
        "calibration_balanced_accuracy": balanced_accuracy,
        "l2": float(l2),
    }


def binary_normal_probability(features: np.ndarray, calibration: dict) -> float:
    """Apply a calibration returned by ``fit_binary_logistic_calibration``."""
    values = np.asarray(features, dtype=np.float64).reshape(-1)
    mean = np.asarray(calibration["mean"], dtype=np.float64)
    scale = np.asarray(calibration["scale"], dtype=np.float64)
    impute = np.asarray(calibration["impute"], dtype=np.float64)
    coefficients = np.asarray(calibration["coefficients"], dtype=np.float64)
    if not (values.size == mean.size == scale.size == impute.size == coefficients.size):
        raise ValueError("binary classifier calibration 차원이 일치하지 않습니다.")
    values = np.where(np.isfinite(values), values, impute)
    normalized = (values - mean) / np.where(scale > 1e-8, scale, 1.0)
    logit = float(calibration["intercept"]) + float(normalized @ coefficients)
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute tie-aware ROC-AUC without an external ML dependency."""
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    n_positive = int(labels.sum())
    n_negative = int((~labels).sum())
    if n_positive == 0 or n_negative == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) * 0.5
        start = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - n_positive * (n_positive + 1) * 0.5) / (
        n_positive * n_negative
    )


def distance_to_unit(d: np.ndarray | float, calib: dict) -> np.ndarray:
    """Map a calibrated distance to a clipped 0..1 distance.

    ``lo`` and ``hi`` are the robust anchors used by ``distance_to_score``.
    Using the same anchors puts DTW and DDTW on a common scale before their
    weighted blend is calculated.
    """
    lo = float(calib["lo"])
    hi = float(calib["hi"])
    if hi <= lo:
        raise ValueError("Distance calibration requires hi > lo")
    values = np.asarray(d, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def fit_hybrid_calibration(
    dtw_normal_distances: np.ndarray,
    ddtw_normal_distances: np.ndarray,
    dtw_error_distances: np.ndarray,
    ddtw_error_distances: np.ndarray,
    *,
    alpha_step: float = 0.01,
) -> dict:
    """Fit normalized ``alpha*DTW + (1-alpha)*DDTW`` calibration.

    Each component first uses its own robust ``lo``/``hi`` anchors.  ``alpha``
    is then selected on a deterministic grid by maximizing the same
    normal/error balanced accuracy used by ``fit_score_calibration``.  The
    returned threshold is on the normalized hybrid-distance scale.
    """
    arrays = [
        np.asarray(dtw_normal_distances, dtype=np.float64),
        np.asarray(ddtw_normal_distances, dtype=np.float64),
        np.asarray(dtw_error_distances, dtype=np.float64),
        np.asarray(ddtw_error_distances, dtype=np.float64),
    ]
    if arrays[0].size != arrays[1].size or arrays[2].size != arrays[3].size:
        raise ValueError("DTW and DDTW calibration arrays must have matching lengths")

    normal_mask = np.isfinite(arrays[0]) & np.isfinite(arrays[1])
    error_mask = np.isfinite(arrays[2]) & np.isfinite(arrays[3])
    if not normal_mask.any() or not error_mask.any():
        raise ValueError("Hybrid calibration requires finite normal and error pairs")
    dtw_normal, ddtw_normal = arrays[0][normal_mask], arrays[1][normal_mask]
    dtw_error, ddtw_error = arrays[2][error_mask], arrays[3][error_mask]

    dtw_calib = fit_score_calibration(dtw_normal, dtw_error)
    ddtw_calib = fit_score_calibration(ddtw_normal, ddtw_error)
    normal_dtw_unit = distance_to_unit(dtw_normal, dtw_calib)
    normal_ddtw_unit = distance_to_unit(ddtw_normal, ddtw_calib)
    error_dtw_unit = distance_to_unit(dtw_error, dtw_calib)
    error_ddtw_unit = distance_to_unit(ddtw_error, ddtw_calib)

    step = float(alpha_step)
    if not (0.0 < step <= 1.0):
        raise ValueError("alpha_step must be in (0, 1]")
    # Include both endpoints even when floating-point stepping would omit 1.
    alpha_values = np.unique(np.concatenate([
        np.arange(0.0, 1.0 + step * 0.5, step),
        np.asarray([0.0, 1.0]),
    ]))
    best: dict | None = None
    alpha_sweep_01: list[dict[str, float]] = []
    for alpha in alpha_values:
        normal_hybrid = alpha * normal_dtw_unit + (1.0 - alpha) * normal_ddtw_unit
        error_hybrid = alpha * error_dtw_unit + (1.0 - alpha) * error_ddtw_unit
        candidate = fit_score_calibration(normal_hybrid, error_hybrid)
        candidate_alpha = float(alpha)
        alpha_index = int(round(candidate_alpha * 100.0))
        if alpha_index % 10 == 0:
            alpha_sweep_01.append(
                {
                    "alpha": round(candidate_alpha, 1),
                    "balanced_accuracy": float(candidate["calibration_balanced_accuracy"]),
                    "normal_distance_threshold": float(candidate["normal_distance_threshold"]),
                }
            )
        if best is None:
            choose = True
        else:
            score_delta = candidate["calibration_balanced_accuracy"] - best["calibration_balanced_accuracy"]
            # On a tie prefer the balanced blend (then the smaller alpha) to
            # avoid an arbitrary endpoint winning because of grid ordering.
            choose = score_delta > 1e-12 or (
                abs(score_delta) <= 1e-12
                and abs(candidate_alpha - 0.5) < abs(float(best["alpha"]) - 0.5)
            )
        if choose:
            best = dict(candidate)
            best["alpha"] = candidate_alpha

    assert best is not None  # alpha_values always contains 0 and 1
    best["algorithm"] = "hybrid"
    best["alpha_step"] = step
    best["alpha_sweep_0_1"] = alpha_sweep_01
    best["components"] = {
        "dtw": dtw_calib,
        "ddtw": ddtw_calib,
    }
    return best
