"""초급·중급·고급 Reference 결과를 하나의 실시간 판정으로 결합한다."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .reference_levels import DIFFICULTY_LEVELS, NORMAL_CLASS, REFERENCE_CLASSES
from .scoring import binary_normal_probability, distance_to_score


INDETERMINATE_CLASS = "판정 불가"
UNSTABLE_CLASS = "동작 인식 불안정"
ERROR_CLASSES = tuple(
    class_label for class_label in REFERENCE_CLASSES if class_label != NORMAL_CLASS
)
BINARY_DISTANCE_FEATURE_NAMES = (
    "normal_min",
    "normal_median",
    "normal_max",
    "normal_spread",
    "error_min",
    "error_median",
    "error_minus_normal",
)


@dataclass(frozen=True)
class ReferenceMatchDecision:
    """세 난이도 Reference를 모두 비교한 최종 판정."""

    predicted_class: str
    matched_level: str
    match_rate: float | None
    normal_match_by_level: dict[str, float | None]
    normal_distance_by_level: dict[str, float]
    raw_distance_by_class: dict[str, float]
    selected_result: dict[str, Any]
    decision_status: str
    rejection_reason: str | None
    error_relative_margin: float | None
    normal_probability: float | None = None
    error_z_margin: float | None = None


def _distance(result: dict[str, Any], method: str) -> float:
    try:
        return float(result[method])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Reference 비교 결과에 유효한 {method!r} 값이 없습니다.") from error


def _finite_summary(values: list[float]) -> tuple[float, float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.inf, np.inf, np.inf, np.inf
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    return minimum, float(np.median(finite)), maximum, maximum - minimum


def binary_distance_features(
    per_level: dict[str, dict[str, dict[str, Any]]],
    method: str = "min_distance",
) -> np.ndarray:
    """Build the fixed-size distance vector used by the first-stage model."""
    normal = [
        _distance(per_level[level][NORMAL_CLASS], method)
        for level in DIFFICULTY_LEVELS
    ]
    errors = [
        min(
            _distance(per_level[level][class_label], method)
            for level in DIFFICULTY_LEVELS
        )
        for class_label in ERROR_CLASSES
    ]
    normal_min, normal_median, normal_max, normal_spread = _finite_summary(normal)
    error_min, error_median, _, _ = _finite_summary(errors)
    margin = (
        error_min - normal_min
        if np.isfinite(error_min) and np.isfinite(normal_min)
        else np.inf
    )
    return np.asarray(
        [
            normal_min,
            normal_median,
            normal_max,
            normal_spread,
            error_min,
            error_median,
            margin,
        ],
        dtype=np.float64,
    )


def decide_reference_match(
    per_level: dict[str, dict[str, dict[str, Any]]],
    *,
    score_calibration: dict | None = None,
    rejection_config: dict | None = None,
    method: str = "min_distance",
    heel_contact_2d: bool | None = None,
) -> ReferenceMatchDecision:
    """세 난이도 중 하나라도 정상과 충분히 유사하면 정상으로 판정한다.

    ``per_level[level][class]``는 ``multi_reference_distance`` 결과다. 보정값이
    있으면 세 정상 distance 중 최솟값을 학습된 threshold와 비교한다. 보정값이
    없는 partial 판정에서는 각 난이도 안에서 정상 distance가 모든 오류 class보다
    엄격히 작을 때만 정상 후보로 인정한다.
    """

    missing_levels = [level for level in DIFFICULTY_LEVELS if level not in per_level]
    if missing_levels:
        raise ValueError(f"Reference 비교 결과에 난이도가 없습니다: {', '.join(missing_levels)}")

    for level in DIFFICULTY_LEVELS:
        missing_classes = [cls for cls in REFERENCE_CLASSES if cls not in per_level[level]]
        if missing_classes:
            raise ValueError(
                f"{level} Reference 비교 결과에 클래스가 없습니다: {', '.join(missing_classes)}"
            )

    normal_distance_by_level = {
        level: _distance(per_level[level][NORMAL_CLASS], method)
        for level in DIFFICULTY_LEVELS
    }
    best_normal_level = min(
        DIFFICULTY_LEVELS, key=lambda level: normal_distance_by_level[level]
    )
    best_normal_distance = normal_distance_by_level[best_normal_level]

    raw_distance_by_class = {
        cls: min(_distance(per_level[level][cls], method) for level in DIFFICULTY_LEVELS)
        for cls in REFERENCE_CLASSES
    }

    rejection_config = rejection_config or {}
    normal_match_by_level: dict[str, float | None]
    normal_probability: float | None = None
    if score_calibration is not None:
        normal_match_by_level = {
            level: distance_to_score(distance, score_calibration)
            for level, distance in normal_distance_by_level.items()
        }
        binary_classifier = score_calibration.get("binary_classifier")
        use_binary_classifier = bool(
            rejection_config.get("use_binary_classifier", True)
        )
        if isinstance(binary_classifier, dict) and use_binary_classifier:
            names = tuple(binary_classifier.get("feature_names", ()))
            if names != BINARY_DISTANCE_FEATURE_NAMES:
                raise ValueError("binary classifier feature schema가 일치하지 않습니다.")
            normal_probability = binary_normal_probability(
                binary_distance_features(per_level, method),
                binary_classifier,
            )
            probability_threshold = float(
                binary_classifier.get("probability_threshold", 0.5)
            )
            is_normal = normal_probability >= probability_threshold
        else:
            threshold = score_calibration.get("normal_distance_threshold")
            if threshold is not None:
                is_normal = best_normal_distance <= float(threshold)
            else:
                match_threshold = float(score_calibration.get("normal_match_threshold", 60.0))
                is_normal = float(normal_match_by_level[best_normal_level]) >= match_threshold
    else:
        normal_match_by_level = {level: None for level in DIFFICULTY_LEVELS}
        normal_levels = []
        for level in DIFFICULTY_LEVELS:
            best_error = min(
                _distance(per_level[level][cls], method)
                for cls in REFERENCE_CLASSES
                if cls != NORMAL_CLASS
            )
            if normal_distance_by_level[level] < best_error:
                normal_levels.append(level)
        is_normal = bool(normal_levels)
        if normal_levels:
            best_normal_level = min(
                normal_levels, key=lambda level: normal_distance_by_level[level]
            )

    # 신뢰 가능한 2D heel 상승은 정상 threshold보다 우선하는 보조 근거다.
    # 반대로 안정적인 heel은 정상 전체 동작을 보장하지 않고 heel 오류만 억제한다.
    if heel_contact_2d is False and bool(
        rejection_config.get("heel_lift_overrides_normal", True)
    ):
        is_normal = False

    decision_status = "normal" if is_normal else "error"
    rejection_reason = None
    error_relative_margin = None
    error_z_margin = None

    if is_normal:
        predicted_class = NORMAL_CLASS
        matched_level = best_normal_level
        selected_result = per_level[matched_level][predicted_class]
    else:
        finite_error_pairs = [
            (level, cls)
            for level in DIFFICULTY_LEVELS
            for cls in REFERENCE_CLASSES
            if cls != NORMAL_CLASS
            and np.isfinite(_distance(per_level[level][cls], method))
        ]
        if not finite_error_pairs or not np.isfinite(best_normal_distance):
            matched_level = best_normal_level
            predicted_class = UNSTABLE_CLASS
            selected_result = per_level[best_normal_level][NORMAL_CLASS]
            decision_status = "unstable"
            rejection_reason = "phase_length_or_warping_invalid"
        else:
            normalization = (
                score_calibration.get("error_distance_normalization", {})
                if score_calibration is not None
                else {}
            )
            use_normalization = bool(
                rejection_config.get("use_error_distance_normalization", True)
            ) and all(class_label in normalization for class_label in ERROR_CLASSES)
            if use_normalization:
                error_ranking = []
                for class_label in ERROR_CLASSES:
                    stats = normalization[class_label]
                    scale = max(float(stats.get("iqr", 1.0)), 1e-8)
                    z_distance = (
                        raw_distance_by_class[class_label]
                        - float(stats.get("median", 0.0))
                    ) / scale
                    error_ranking.append((z_distance, class_label))
                error_ranking.sort(key=lambda item: item[0])
                closest_error_class = error_ranking[0][1]
                if len(error_ranking) >= 2:
                    error_z_margin = float(
                        error_ranking[1][0] - error_ranking[0][0]
                    )
            else:
                closest_error_class = min(
                    ERROR_CLASSES,
                    key=lambda class_label: raw_distance_by_class[class_label],
                )

            heel_lift_forced = heel_contact_2d is False and bool(
                rejection_config.get("heel_lift_selects_heel_error", True)
            )
            if heel_lift_forced:
                closest_error_class = ERROR_CLASSES[0]
                error_z_margin = None

            matched_level = min(
                DIFFICULTY_LEVELS,
                key=lambda level: _distance(
                    per_level[level][closest_error_class], method
                ),
            )
            selected_result = per_level[matched_level][closest_error_class]
            predicted_class = closest_error_class
            selected_error_distance = raw_distance_by_class[closest_error_class]

            class_error_distances = sorted(
                (
                    (raw_distance_by_class[cls], cls)
                    for cls in REFERENCE_CLASSES
                    if cls != NORMAL_CLASS and np.isfinite(raw_distance_by_class[cls])
                ),
                key=lambda item: item[0],
            )
            best_error_distance = class_error_distances[0][0]
            if len(class_error_distances) >= 2:
                second_error_distance = class_error_distances[1][0]
                error_relative_margin = (
                    second_error_distance - best_error_distance
                ) / max(abs(best_error_distance), 1e-8)

            if score_calibration is not None:
                thresholds = score_calibration.get("error_distance_thresholds", {})
                error_threshold = thresholds.get(closest_error_class)
                if error_threshold is None:
                    normal_threshold = score_calibration.get("normal_distance_threshold")
                    if normal_threshold is not None:
                        scale = float(
                            rejection_config.get("error_distance_scale_from_normal", 1.5)
                        )
                        error_threshold = float(normal_threshold) * scale
                min_margin = float(
                    score_calibration.get(
                        "min_error_relative_margin",
                        rejection_config.get("min_error_relative_margin", 0.05),
                    )
                )
                min_z_margin = score_calibration.get("min_error_z_margin")
                if error_threshold is not None and selected_error_distance > float(error_threshold):
                    predicted_class = UNSTABLE_CLASS
                    decision_status = "unstable"
                    rejection_reason = "all_error_references_too_far"
                elif (
                    use_normalization
                    and not heel_lift_forced
                    and min_z_margin is not None
                    and error_z_margin is not None
                    and error_z_margin < float(min_z_margin)
                ):
                    predicted_class = INDETERMINATE_CLASS
                    decision_status = "indeterminate"
                    rejection_reason = "normalized_error_classes_too_close"
                elif (
                    not use_normalization
                    and error_relative_margin is not None
                    and error_relative_margin < min_margin
                ):
                    predicted_class = INDETERMINATE_CLASS
                    decision_status = "indeterminate"
                    rejection_reason = "error_classes_too_close"

            if (
                predicted_class == ERROR_CLASSES[0]
                and heel_contact_2d is True
                and bool(rejection_config.get("stable_heel_suppresses_heel_error", True))
            ):
                predicted_class = INDETERMINATE_CLASS
                decision_status = "indeterminate"
                rejection_reason = "heel_error_not_supported_by_2d"

    match_rate = (
        100.0 * normal_probability
        if normal_probability is not None
        else None
        if normal_match_by_level[best_normal_level] is None
        else float(normal_match_by_level[best_normal_level])
    )
    return ReferenceMatchDecision(
        predicted_class=predicted_class,
        matched_level=matched_level,
        match_rate=match_rate,
        normal_match_by_level=normal_match_by_level,
        normal_distance_by_level=normal_distance_by_level,
        raw_distance_by_class=raw_distance_by_class,
        selected_result=selected_result,
        decision_status=decision_status,
        rejection_reason=rejection_reason,
        error_relative_margin=error_relative_margin,
        normal_probability=normal_probability,
        error_z_margin=error_z_margin,
    )


__all__ = [
    "INDETERMINATE_CLASS",
    "UNSTABLE_CLASS",
    "ReferenceMatchDecision",
    "BINARY_DISTANCE_FEATURE_NAMES",
    "binary_distance_features",
    "decide_reference_match",
]
