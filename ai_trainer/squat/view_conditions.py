"""View-specific, dataset-calibrated squat condition features and evaluator.

This module deliberately does not replace the sequence classifier.  It produces
interpretable evidence which can confirm or veto a model decision.  Thresholds
are generated from actor-disjoint AI Hub training data by
``scripts/build_view_conditions.py``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ai_trainer.squat.camera_views import VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT, VIEWS
from ai_trainer.core.common_skeleton import COMMON_JOINT_NAMES

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "view_condition_thresholds.json"
PHASES = ("준비", "하강", "최저점", "상승", "종료")
_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    first = a - b
    second = c - b
    denom = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    cosine = np.divide(
        np.sum(first * second, axis=-1),
        denom,
        out=np.zeros_like(denom),
        where=denom > np.finfo(np.float64).eps,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _line_tilt_deg(vector: np.ndarray, reference: str) -> np.ndarray:
    dx, dy = vector[:, 0], vector[:, 1]
    if reference == "vertical":
        return np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    return np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))


def _leg_scale(coords: np.ndarray) -> np.ndarray:
    lengths = []
    for side in ("L", "R"):
        hip = coords[:, _IDX[f"{side}Hip"]]
        knee = coords[:, _IDX[f"{side}Knee"]]
        ankle = coords[:, _IDX[f"{side}Ankle"]]
        lengths.append(np.linalg.norm(hip - knee, axis=-1) + np.linalg.norm(knee - ankle, axis=-1))
    return np.maximum(np.mean(lengths, axis=0), 1e-6)


def extract_view_features(coords_2d: np.ndarray, view: str) -> dict[str, np.ndarray]:
    """Extract scale-independent 2D features for one camera view.

    ``coords_2d`` uses the 18-joint common skeleton and may be in pixels or any
    uniformly scaled image coordinate system.
    """
    if view not in VIEWS:
        raise ValueError(f"지원하지 않는 시점입니다: {view}")
    coords = np.asarray(coords_2d, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[1:] != (len(COMMON_JOINT_NAMES), 2):
        raise ValueError(f"coords_2d shape은 (T,18,2)여야 합니다: {coords.shape}")
    scale = _leg_scale(coords)

    lhip, rhip = coords[:, _IDX["LHip"]], coords[:, _IDX["RHip"]]
    lknee, rknee = coords[:, _IDX["LKnee"]], coords[:, _IDX["RKnee"]]
    lankle, rankle = coords[:, _IDX["LAnkle"]], coords[:, _IDX["RAnkle"]]
    ltoe, rtoe = coords[:, _IDX["LBigToe"]], coords[:, _IDX["RBigToe"]]
    lheel, rheel = coords[:, _IDX["LHeel"]], coords[:, _IDX["RHeel"]]
    lshoulder, rshoulder = coords[:, _IDX["LShoulder"]], coords[:, _IDX["RShoulder"]]
    hip_center, neck = coords[:, _IDX["Hip"]], coords[:, _IDX["Neck"]]
    common_features = {
        # View/framing evidence.  These are calibrated from the same cameras as
        # the scoring conditions but are deliberately excluded from error-rule
        # selection: turning the body is not a squat-form error.
        "hip_screen_separation_ratio": np.linalg.norm(lhip - rhip, axis=-1) / scale,
        "shoulder_screen_separation_ratio": np.linalg.norm(lshoulder - rshoulder, axis=-1) / scale,
    }

    if view == VIEW_FRONT:
        left_knee_deviation = np.abs(180.0 - _angle_deg(lhip, lknee, lankle))
        right_knee_deviation = np.abs(180.0 - _angle_deg(rhip, rknee, rankle))
        body_center_x = hip_center[:, 0]
        left_inward = np.sign(body_center_x - ltoe[:, 0])
        right_inward = np.sign(body_center_x - rtoe[:, 0])
        left_medial = (lknee[:, 0] - ltoe[:, 0]) * left_inward / scale
        right_medial = (rknee[:, 0] - rtoe[:, 0]) * right_inward / scale
        return common_features | {
            "left_knee_deviation_deg": left_knee_deviation,
            "right_knee_deviation_deg": right_knee_deviation,
            "knee_deviation_mean_deg": (left_knee_deviation + right_knee_deviation) / 2.0,
            "knee_deviation_asymmetry_deg": np.abs(left_knee_deviation - right_knee_deviation),
            "left_knee_medial_ratio": left_medial,
            "right_knee_medial_ratio": right_medial,
            "knee_medial_max_ratio": np.maximum(left_medial, right_medial),
            "pelvis_tilt_deg": _line_tilt_deg(lhip - rhip, "horizontal"),
            "shoulder_tilt_deg": _line_tilt_deg(lshoulder - rshoulder, "horizontal"),
            "trunk_lateral_lean_deg": _line_tilt_deg(neck - hip_center, "vertical"),
            "pelvis_lateral_shift_ratio": np.abs(hip_center[:, 0] - (lankle[:, 0] + rankle[:, 0]) / 2.0) / scale,
            "stance_width_ratio": np.abs(lankle[:, 0] - rankle[:, 0]) / scale,
            "knee_height_asymmetry_ratio": np.abs(lknee[:, 1] - rknee[:, 1]) / scale,
            "hip_height_asymmetry_ratio": np.abs(lhip[:, 1] - rhip[:, 1]) / scale,
        }

    side = "L" if view == VIEW_LEFT else "R"
    shoulder = coords[:, _IDX[f"{side}Shoulder"]]
    hip = coords[:, _IDX[f"{side}Hip"]]
    knee = coords[:, _IDX[f"{side}Knee"]]
    ankle = coords[:, _IDX[f"{side}Ankle"]]
    heel = coords[:, _IDX[f"{side}Heel"]]
    toe = coords[:, _IDX[f"{side}BigToe"]]
    foot_vector = toe - heel
    forward_sign = np.sign(foot_vector[:, 0])
    forward_sign[forward_sign == 0] = 1.0
    knee_angle = _angle_deg(hip, knee, ankle)
    hip_angle = _angle_deg(shoulder, hip, knee)
    ankle_angle = _angle_deg(knee, ankle, toe)
    return common_features | {
        "visible_knee_angle_deg": knee_angle,
        "visible_knee_flexion_deg": 180.0 - knee_angle,
        "visible_hip_angle_deg": hip_angle,
        "visible_hip_flexion_deg": 180.0 - hip_angle,
        "visible_ankle_angle_deg": ankle_angle,
        "trunk_forward_lean_deg": _line_tilt_deg(neck - hip_center, "vertical"),
        "visible_thigh_tilt_deg": _line_tilt_deg(knee - hip, "horizontal"),
        "hip_to_knee_vertical_ratio": (hip[:, 1] - knee[:, 1]) / scale,
        "knee_forward_to_toe_ratio": (knee[:, 0] - toe[:, 0]) * forward_sign / scale,
        "heel_lift_ratio": (toe[:, 1] - heel[:, 1]) / scale,
        "foot_tilt_deg": _line_tilt_deg(foot_vector, "horizontal"),
    }


def summarize_view_features(
    features: dict[str, np.ndarray], phase_bounds: dict[str, list[int] | tuple[int, int]]
) -> dict[str, float]:
    """Summarize a repetition into phase/feature/stat scalar conditions."""
    if not features:
        return {}
    n_frames = len(next(iter(features.values())))
    out: dict[str, float] = {}
    for phase in PHASES:
        start, end = phase_bounds.get(phase, (0, 0))
        start = max(0, min(int(start), n_frames))
        end = max(start, min(int(end), n_frames))
        out[f"timing|{phase}|ratio"] = float((end - start) / max(n_frames, 1))
        if end - start < 1:
            continue
        for name, values in features.items():
            segment = np.asarray(values[start:end], dtype=np.float64)
            finite = segment[np.isfinite(segment)]
            if finite.size == 0:
                continue
            prefix = f"{phase}|{name}"
            out[f"{prefix}|median"] = float(np.median(finite))
            out[f"{prefix}|p10"] = float(np.percentile(finite, 10))
            out[f"{prefix}|p90"] = float(np.percentile(finite, 90))
            out[f"{prefix}|range"] = float(np.percentile(finite, 90) - np.percentile(finite, 10))
    return out


@dataclass(frozen=True)
class RuleHit:
    error_class: str
    metric: str
    value: float
    operator: str
    threshold: float
    validation_balanced_accuracy: float


@dataclass(frozen=True)
class PaperPostureViolation:
    """One failed lowest-position criterion from the cited squat paper.

    Values are expressed either in degrees or in image distance divided by the
    visible leg length.  The latter makes the paper's pixel-coordinate rules
    independent of the user's camera distance and image resolution.
    """

    condition: str
    value: float
    lower: float | None
    upper: float | None
    message: str


@dataclass(frozen=True)
class ViewConditionAssessment:
    view: str
    normal_coverage: float
    error_support: dict[str, float]
    predicted_error: str | None
    rule_hits: tuple[RuleHit, ...]
    paper_posture_violations: tuple[PaperPostureViolation, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["rule_hits"] = [asdict(hit) for hit in self.rule_hits]
        payload["paper_posture_violations"] = [
            asdict(violation) for violation in self.paper_posture_violations
        ]
        return payload


def assess_paper_squat_conditions(
    coords_2d: np.ndarray,
    phase_bounds: dict[str, list[int] | tuple[int, int]],
    view: str,
) -> tuple[PaperPostureViolation, ...]:
    """Evaluate the three lowest-position criteria in the cited paper.

    ``Real-Time Workout Posture Correction using OpenCV and MediaPipe``
    defines a correct squat at its lowest position as: hip angle 60--120°, a
    nearly horizontal thigh (hip/knee y difference <= 0.2), and a knee/toe
    x difference <= 0.1.  The last two were originally written for image
    coordinates.  Here they are divided by the camera-near leg length, so the
    same numerical limits remain meaningful at different camera distances.

    These are side-view-only criteria.  A frontal image cannot reliably tell
    whether the knee is forward of the toe or whether the thigh is horizontal
    in the sagittal plane.  When the phase state machine records a zero-frame
    bottom during an immediate descent-to-ascent reversal, the deepest visible
    knee frame is used instead.
    """
    if view not in VIEWS:
        raise ValueError(f"지원하지 않는 시점입니다: {view}")
    if view == VIEW_FRONT:
        return ()

    coords = np.asarray(coords_2d, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[1:] != (len(COMMON_JOINT_NAMES), 2):
        raise ValueError(f"coords_2d shape은 (T,18,2)여야 합니다: {coords.shape}")
    if len(coords) == 0 or not np.isfinite(coords).all():
        return ()

    side = "L" if view == VIEW_LEFT else "R"
    shoulder = coords[:, _IDX[f"{side}Shoulder"]]
    hip = coords[:, _IDX[f"{side}Hip"]]
    knee = coords[:, _IDX[f"{side}Knee"]]
    ankle = coords[:, _IDX[f"{side}Ankle"]]
    toe = coords[:, _IDX[f"{side}BigToe"]]
    visible_leg_scale = np.maximum(
        np.linalg.norm(hip - knee, axis=-1) + np.linalg.norm(knee - ankle, axis=-1),
        1e-6,
    )
    hip_angle = _angle_deg(knee, hip, shoulder)

    start, end = phase_bounds.get("최저점", (0, 0))
    start = max(0, min(int(start), len(coords)))
    end = max(start, min(int(end), len(coords)))
    if end <= start:
        # The deepest knee bend is a robust single-frame proxy for the bottom.
        center = int(np.argmin(_angle_deg(hip, knee, ankle)))
        start, end = max(0, center - 1), min(len(coords), center + 2)

    selection = slice(start, end)
    hip_angle_value = float(np.median(hip_angle[selection]))
    thigh_height_value = float(np.median(
        np.abs(hip[selection, 1] - knee[selection, 1]) / visible_leg_scale[selection]
    ))
    knee_toe_value = float(np.median(
        np.abs(knee[selection, 0] - toe[selection, 0]) / visible_leg_scale[selection]
    ))

    violations: list[PaperPostureViolation] = []
    if hip_angle_value < 60.0:
        violations.append(PaperPostureViolation(
            condition="조건 1 · 고관절 각도",
            value=hip_angle_value,
            lower=60.0,
            upper=120.0,
            message=(f"엉덩이를 더 높게 하세요 (고관절 각도 {hip_angle_value:.1f}°, "
                     "기준 60–120°)"),
        ))
    elif hip_angle_value > 120.0:
        violations.append(PaperPostureViolation(
            condition="조건 1 · 고관절 각도",
            value=hip_angle_value,
            lower=60.0,
            upper=120.0,
            message=(f"엉덩이를 더 낮게 하세요 (고관절 각도 {hip_angle_value:.1f}°, "
                     "기준 60–120°)"),
        ))
    if thigh_height_value > 0.20:
        violations.append(PaperPostureViolation(
            condition="조건 2 · 허벅지 수평",
            value=thigh_height_value,
            lower=None,
            upper=0.20,
            message=(f"허벅지를 바닥과 수평으로 유지하세요 (허벅지 높이 차 "
                     f"{thigh_height_value:.2f}, 기준 ≤0.20)"),
        ))
    if knee_toe_value > 0.10:
        violations.append(PaperPostureViolation(
            condition="조건 3 · 무릎-발끝 정렬",
            value=knee_toe_value,
            lower=None,
            upper=0.10,
            message=(f"무릎이 발끝을 넘지 않게 하세요 (무릎-발끝 차 "
                     f"{knee_toe_value:.2f}, 기준 ≤0.10)"),
        ))
    return tuple(violations)


def load_view_condition_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assess_view_rep(
    coords_2d: np.ndarray,
    phase_bounds: dict[str, list[int] | tuple[int, int]],
    view: str,
    config: dict | None = None,
) -> ViewConditionAssessment:
    """Evaluate interpretable condition evidence for a completed repetition."""
    cfg = config if config is not None else load_view_condition_config()
    summary = summarize_view_features(extract_view_features(coords_2d, view), phase_bounds)

    ranges = cfg.get("normal_ranges", {}).get(view, {})
    in_range = 0
    checked = 0
    for metric, bounds in ranges.items():
        if metric not in summary:
            continue
        checked += 1
        value = summary[metric]
        if bounds["hard_lower"] <= value <= bounds["hard_upper"]:
            in_range += 1
    normal_coverage = float(in_range / checked) if checked else 0.0

    support: dict[str, float] = {}
    hits: list[RuleHit] = []
    responsibility = cfg.get("responsible_views_by_error", {})
    for error_class, rules in cfg.get("error_rules", {}).get(view, {}).items():
        allowed_views = responsibility.get(error_class)
        if allowed_views is not None and view not in allowed_views:
            support[error_class] = 0.0
            continue
        total_weight = 0.0
        hit_weight = 0.0
        for rule in rules:
            metric = rule["metric"]
            if metric not in summary:
                continue
            # Runtime weighting must not consume the final validation holdout.
            # Schema-v2 stores an actor-disjoint tuning score; validation is
            # reporting-only.  The fallback keeps old schema-v1 configs usable.
            selection_score = rule.get(
                "tuning_balanced_accuracy",
                rule.get("validation_balanced_accuracy", 0.5),
            )
            weight = max(float(selection_score) - 0.5, 0.01)
            total_weight += weight
            value = summary[metric]
            matched = value >= rule["threshold"] if rule["operator"] == ">=" else value <= rule["threshold"]
            if matched:
                hit_weight += weight
                hits.append(
                    RuleHit(
                        error_class=error_class,
                        metric=metric,
                        value=float(value),
                        operator=rule["operator"],
                        threshold=float(rule["threshold"]),
                        validation_balanced_accuracy=float(rule.get("validation_balanced_accuracy", 0.5)),
                    )
                )
        support[error_class] = float(hit_weight / total_weight) if total_weight else 0.0
    predicted = max(support, key=support.get) if support and max(support.values()) >= 0.60 else None
    paper_violations = assess_paper_squat_conditions(coords_2d, phase_bounds, view)
    return ViewConditionAssessment(
        view=view,
        normal_coverage=normal_coverage,
        error_support=support,
        predicted_error=predicted,
        rule_hits=tuple(sorted(hits, key=lambda item: -item.validation_balanced_accuracy)),
        paper_posture_violations=paper_violations,
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PHASES",
    "RuleHit",
    "PaperPostureViolation",
    "ViewConditionAssessment",
    "assess_view_rep",
    "assess_paper_squat_conditions",
    "extract_view_features",
    "load_view_condition_config",
    "summarize_view_features",
]
