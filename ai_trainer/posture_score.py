"""Normal-reference posture match for completed REPs and lightweight live display."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES
from .dtw_compare import _dtw_dp
from .lifting_dataset import WINDOW_T
from .two_d_diagnostic import extract_2d_features

I = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
REQUIRED_VISIBILITY = (
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
    "LHeel", "RHeel", "LFootIndex", "RFootIndex",
)
COMPONENTS = ("depth", "hip", "knee", "heel_stability", "balance", "trajectory")
PHASE_NAMES = {"prep": "준비", "descend": "하강", "bottom": "최저점", "ascend": "상승"}
REALTIME_EMA_ALPHA = 0.25
MATCH_VERSION = 3
MATCH_METHOD = "hybrid_2d3d_robust_joint_scale_v2"


def _stats(values) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p10", "median", "p90", "max"),
            np.percentile(values, (0, 10, 50, 90, 100)),
        )
    }


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cost = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    total, path_length = _dtw_dp(cost, window_ratio=None)
    return float(total / max(path_length, 1))


def _nearest_distance(query, references, distance_fn) -> float:
    return min(float(distance_fn(query, reference)) for reference in references)


def _intra_distances(references, distance_fn) -> list[float]:
    return [
        min(float(distance_fn(reference, other)) for j, other in enumerate(references) if i != j)
        for i, reference in enumerate(references)
    ]


def _ratio_score(distance: float, normal_scale: float) -> float:
    """100 within normal median nearest-neighbour deviation, then decay by ratio."""
    scale = max(float(normal_scale), 1e-8)
    return float(np.clip(100.0 * scale / max(scale, float(distance)), 0.0, 100.0))


def _vector_metric(references: list[np.ndarray]):
    stack = np.stack(references)
    spread = np.percentile(stack, 90, axis=0) - np.percentile(stack, 10, axis=0)
    fallback = np.maximum(np.abs(np.median(stack, axis=0)) * 0.05, 1e-6)
    scale = np.maximum(spread, fallback)
    return lambda a, b: float(np.linalg.norm((np.asarray(a) - np.asarray(b)) / scale))


def _depth_3d(features: dict[str, np.ndarray]) -> np.ndarray:
    """Hip/knee excursion in degrees, entirely in the lifting-3D domain."""
    hip = np.asarray(features["hip_flexion_angle"], dtype=float).mean(axis=1) * 180.0
    knee = np.asarray(features["knee_flexion_angle"], dtype=float).mean(axis=1) * 180.0
    head = min(5, len(hip))
    return np.asarray([
        float(np.median(knee[:head]) - np.min(knee)),
        float(np.median(hip[:head]) - np.min(hip)),
    ])


def _balance_2d(points: np.ndarray) -> np.ndarray:
    """Projection-stable 2D balance trajectory without raw joint-angle asymmetry."""
    p = np.asarray(points, dtype=float)
    pelvis = (p[:, I["LHip"]] + p[:, I["RHip"]]) / 2.0
    shoulders = (p[:, I["LShoulder"]] + p[:, I["RShoulder"]]) / 2.0
    ankles = (p[:, I["LAnkle"]] + p[:, I["RAnkle"]]) / 2.0
    torso = np.linalg.norm(shoulders - pelvis, axis=1)
    scale = max(float(np.median(torso[:min(5, len(torso))])), 1e-6)
    origin = np.median(pelvis[:min(5, len(pelvis))], axis=0)
    # Centers and relative displacements are much less sensitive to frontal-view
    # left/right angle collapse than comparing left and right angles directly.
    return np.column_stack([
        (pelvis[:, 0] - origin[0]) / scale,
        (shoulders[:, 0] - pelvis[:, 0]) / scale,
        (ankles[:, 0] - pelvis[:, 0]) / scale,
        np.linalg.norm(shoulders - pelvis, axis=1) / scale,
    ])


@dataclass(frozen=True)
class PostureScoreResult:
    score_valid: bool
    overall: float | None
    components: dict[str, float | None]
    measurement_valid: bool
    invalid_reason: str | None
    calculation_ms: float
    reference_calibration: dict
    diagnostic: dict

    def as_dict(self) -> dict:
        return {
            # Legacy keys are retained so existing session records remain readable.
            "score_valid": self.score_valid,
            "overall": self.overall,
            "match_valid": self.score_valid,
            "overall_match": self.overall,
            "match_version": MATCH_VERSION,
            "match_method": MATCH_METHOD,
            **self.components,
            "measurement_valid": self.measurement_valid,
            "invalid_reason": self.invalid_reason,
            "calculation_ms": self.calculation_ms,
            "diagnostic": self.diagnostic,
        }


class PostureScorer:
    """Scores immutable completed REP copies independently of production classes."""

    def __init__(self, normal_3d_refs: list[dict], normal_2d_refs: list[dict], output_dir=None,
                 *, enable_ab_diagnostic: bool = False):
        self.normal_3d_refs = list(normal_3d_refs)
        self.normal_2d_refs = list(normal_2d_refs)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.enable_ab_diagnostic = bool(enable_ab_diagnostic)
        self._stream = None
        self.reference_calibration = self._build_calibration()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.path = self.output_dir / f"posture_score_{stamp}.jsonl"
            self._stream = self.path.open("x", encoding="utf-8", buffering=1)
            self._write({"record_type": "reference_calibration", **self.reference_calibration})

    def _build_calibration(self) -> dict:
        if len(self.normal_2d_refs) < 2 or len(self.normal_3d_refs) < 2:
            return {"valid": False, "reason": "at least two normal 2D and 3D references are required"}
        legacy_depth_refs = [np.array([r["summary"]["knee_excursion"], r["summary"]["hip_excursion"]]) for r in self.normal_2d_refs]
        depth_refs = [_depth_3d(r["feat"]) for r in self.normal_3d_refs]
        heel_refs = [np.array([
            r["summary"]["left_heel_ankle_motion"], r["summary"]["right_heel_ankle_motion"],
            r["summary"]["left_heel_toe_delta"], r["summary"]["right_heel_toe_delta"],
        ]) for r in self.normal_2d_refs]
        sequences = {
            "hip": [np.asarray(r["feat"]["hip_flexion_angle"]) for r in self.normal_3d_refs],
            "knee": [np.asarray(r["feat"]["knee_flexion_angle"]) for r in self.normal_3d_refs],
            "balance": [_balance_2d(np.asarray(r["raw"])) for r in self.normal_2d_refs],
            "trajectory": [np.asarray(r["feat"]["joint_coords_3d"]) for r in self.normal_3d_refs],
        }
        refs = {"depth": depth_refs, "heel_stability": heel_refs, **sequences}
        metrics = {"depth": _vector_metric(depth_refs), "heel_stability": _vector_metric(heel_refs),
                   "hip": _dtw_distance, "knee": _dtw_distance,
                   "balance": _dtw_distance, "trajectory": _dtw_distance}
        calibration = {"valid": True, "normal_reference_count_2d": len(self.normal_2d_refs),
                       "normal_reference_count_3d": len(self.normal_3d_refs), "components": {}}
        for name in COMPONENTS:
            intra = _intra_distances(refs[name], metrics[name])
            intra_stats = _stats(intra)
            scale_basis = "p90" if name in ("hip", "knee") else "median"
            calibration["components"][name] = {
                "intra_nearest_distance": intra_stats,
                "score_scale": float(intra_stats[scale_basis]),
                "score_scale_basis": f"normal leave-one-out nearest-distance {scale_basis}",
            }
        calibration["components"]["depth"]["feature_distribution"] = {
            "knee_excursion_3d": _stats([v[0] for v in depth_refs]),
            "hip_excursion_3d": _stats([v[1] for v in depth_refs]),
        }
        calibration["components"]["heel_stability"]["feature_distribution"] = {
            "heel_relative_motion": _stats([max(v[0], v[1]) for v in heel_refs])
        }
        calibration["match_version"] = MATCH_VERSION
        calibration["match_method"] = MATCH_METHOD
        calibration["component_sources"] = {
            "depth": "lifting_3d", "hip": "lifting_3d", "knee": "lifting_3d",
            "heel_stability": "mediapipe_2d", "balance": "mediapipe_2d",
            "trajectory": "lifting_3d",
        }
        calibration["legacy_depth_feature_distribution"] = {
            "knee_excursion_2d": _stats([v[0] for v in legacy_depth_refs]),
            "hip_excursion_2d": _stats([v[1] for v in legacy_depth_refs]),
        }
        return calibration

    def score(self, raw2d, coords3d, visibility, *, baseline_knee, baseline_hip,
              production_raw=None, production_final=None, rep_index=None) -> PostureScoreResult:
        started = time.perf_counter()
        raw, coords = np.asarray(raw2d, dtype=float), np.asarray(coords3d, dtype=float)
        reason = self._invalid_reason(raw, coords, visibility)
        components = {name: None for name in COMPONENTS}
        if reason is not None or not self.reference_calibration.get("valid", False):
            reason = reason or self.reference_calibration.get("reason", "reference calibration unavailable")
            result = PostureScoreResult(False, None, components, False, reason,
                                        (time.perf_counter() - started) * 1000.0,
                                        self.reference_calibration, {})
        else:
            feat2d, summary = extract_2d_features(raw, baseline_knee=baseline_knee, baseline_hip=baseline_hip)
            from .features import extract_all_features
            feat3d = extract_all_features(coords)
            heel_query = np.array([
                summary["left_heel_ankle_motion"], summary["right_heel_ankle_motion"],
                summary["left_heel_toe_delta"], summary["right_heel_toe_delta"],
            ])
            legacy_queries = {
                "depth": np.array([summary["knee_excursion"], summary["hip_excursion"]]),
                "heel_stability": heel_query,
                "hip": feat2d[:, 2:4], "knee": feat2d[:, 0:2],
                "balance": feat3d["left_right_asymmetry"],
                "trajectory": feat3d["joint_coords_3d"],
            }
            legacy_refs = {
                "depth": [np.array([r["summary"]["knee_excursion"], r["summary"]["hip_excursion"]]) for r in self.normal_2d_refs],
                "heel_stability": [np.array([
                    r["summary"]["left_heel_ankle_motion"], r["summary"]["right_heel_ankle_motion"],
                    r["summary"]["left_heel_toe_delta"], r["summary"]["right_heel_toe_delta"],
                ]) for r in self.normal_2d_refs],
                "hip": [np.asarray(r["feat"])[:, 2:4] for r in self.normal_2d_refs],
                "knee": [np.asarray(r["feat"])[:, 0:2] for r in self.normal_2d_refs],
                "balance": [np.asarray(r["feat"]["left_right_asymmetry"]) for r in self.normal_3d_refs],
                "trajectory": [np.asarray(r["feat"]["joint_coords_3d"]) for r in self.normal_3d_refs],
            }
            legacy_metrics = {"depth": _vector_metric(legacy_refs["depth"]),
                       "heel_stability": _vector_metric(legacy_refs["heel_stability"]),
                       "hip": _dtw_distance, "knee": _dtw_distance,
                       "balance": _dtw_distance, "trajectory": _dtw_distance}
            legacy_components = legacy_deviations = None
            if self.enable_ab_diagnostic:
                legacy_components, legacy_deviations = self._score_components(
                    legacy_queries, legacy_refs, legacy_metrics,
                    calibration_override=self._legacy_calibration(legacy_refs, legacy_metrics),
                )
            queries = {
                "depth": _depth_3d(feat3d),
                "heel_stability": heel_query,
                "hip": feat3d["hip_flexion_angle"],
                "knee": feat3d["knee_flexion_angle"],
                "balance": _balance_2d(raw),
                "trajectory": feat3d["joint_coords_3d"],
            }
            refs = {
                "depth": [_depth_3d(r["feat"]) for r in self.normal_3d_refs],
                "heel_stability": legacy_refs["heel_stability"],
                "hip": [np.asarray(r["feat"]["hip_flexion_angle"]) for r in self.normal_3d_refs],
                "knee": [np.asarray(r["feat"]["knee_flexion_angle"]) for r in self.normal_3d_refs],
                "balance": [_balance_2d(np.asarray(r["raw"])) for r in self.normal_2d_refs],
                "trajectory": legacy_refs["trajectory"],
            }
            metrics = {"depth": _vector_metric(refs["depth"]),
                       "heel_stability": _vector_metric(refs["heel_stability"]),
                       "hip": _dtw_distance, "knee": _dtw_distance,
                       "balance": _dtw_distance, "trajectory": _dtw_distance}
            components, deviations = self._score_components(queries, refs, metrics)
            overall = float(np.mean(list(components.values())))
            depth_distribution = self.reference_calibration["components"]["depth"]["feature_distribution"]
            diagnostic = {
                "hybrid": {"overall": overall, "components": dict(components),
                           "component_distances": deviations},
                "ab_diagnostic_enabled": self.enable_ab_diagnostic,
                "component_sources": dict(self.reference_calibration["component_sources"]),
                "knee_excursion_3d": float(queries["depth"][0]),
                "knee_excursion_normal_3d_median": float(depth_distribution["knee_excursion_3d"]["median"]),
                "hip_excursion_3d": float(queries["depth"][1]),
                "hip_excursion_normal_3d_median": float(depth_distribution["hip_excursion_3d"]["median"]),
                "raw_match": float(overall), "mapped_match": float(overall),
                "mapping": "reference leave-one-out median ratio (unchanged)",
            }
            if self.enable_ab_diagnostic:
                diagnostic["legacy"] = {
                    "overall": float(np.mean(list(legacy_components.values()))),
                    "components": legacy_components,
                    "component_distances": legacy_deviations,
                }
            result = PostureScoreResult(True, overall, components, True, None,
                                        (time.perf_counter() - started) * 1000.0,
                                        self.reference_calibration, diagnostic)
        self._write({"record_type": "completed_rep", "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                     "rep": rep_index, "production_raw": production_raw,
                     "production_final": production_final,
                     "classification_unknown": production_final == "자세추정불확실",
                     **result.as_dict()})
        if result.score_valid:
            print(f"\n[POSTURE MATCH] REP: {rep_index}", flush=True)
            print(f"Production RAW: {production_raw}", flush=True)
            print(f"Production FINAL: {production_final}", flush=True)
            if self.enable_ab_diagnostic:
                legacy = result.diagnostic["legacy"]
                print(f"Legacy overall: {legacy['overall']:.2f}", flush=True)
                print("Legacy components: " + ", ".join(
                    f"{name}={legacy['components'][name]:.2f}" for name in COMPONENTS
                ), flush=True)
            print(f"Hybrid overall: {result.overall:.2f}", flush=True)
            print("Hybrid components: " + ", ".join(
                f"{name}={result.components[name]:.2f}" for name in COMPONENTS
            ), flush=True)
            print("Sources: depth=3D, hip=3D, knee=3D, heel_stability=2D, "
                  "balance=2D, trajectory=3D", flush=True)
        return result

    def _score_components(self, queries, refs, metrics, calibration_override=None):
        calibration = calibration_override or self.reference_calibration["components"]
        scores, deviations = {}, {}
        for name in COMPONENTS:
            deviation = _nearest_distance(queries[name], refs[name], metrics[name])
            scale = calibration[name]["score_scale"]
            scores[name] = _ratio_score(deviation, scale)
            deviations[name] = {"live_distance": float(deviation),
                                "normal_calibration_distance": float(scale),
                                "distance_ratio": float(deviation / max(scale, 1e-8))}
        return scores, deviations

    @staticmethod
    def _legacy_calibration(refs, metrics):
        return {name: {"score_scale": float(np.median(_intra_distances(refs[name], metrics[name])))}
                for name in COMPONENTS}

    @staticmethod
    def _invalid_reason(raw, coords, visibility) -> str | None:
        if raw.ndim != 3 or raw.shape[1:] != (18, 2) or coords.ndim != 3 or coords.shape[1:] != (18, 3):
            return "invalid sequence shape"
        if len(raw) < WINDOW_T or len(coords) < WINDOW_T:
            return f"completed REP is shorter than {WINDOW_T} frames"
        if not np.isfinite(raw).all() or not np.isfinite(coords).all():
            return "required pose coordinates are non-finite"
        if not visibility:
            return "landmark visibility is unavailable"
        for name in REQUIRED_VISIBILITY:
            values = [row.get(name) for row in visibility if row.get(name) is not None]
            if not values or float(np.median(values)) < 0.4:
                return f"insufficient landmark visibility: {name}"
        return None

    def _write(self, row) -> None:
        if self._stream is not None:
            self._stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _single_frame_2d_features(points: np.ndarray) -> np.ndarray:
    """Small 2D feature vector used only for the live match display."""
    p = np.asarray(points, dtype=float)

    def angle(a, b, c):
        u, v = a - b, c - b
        return np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8), -1, 1)))

    knee = [angle(p[I[f"{side}Hip"]], p[I[f"{side}Knee"]], p[I[f"{side}Ankle"]]) / 180.0 for side in ("L", "R")]
    hip = [angle(p[I["Neck"]], p[I[f"{side}Hip"]], p[I[f"{side}Knee"]]) / 180.0 for side in ("L", "R")]
    torso = max(float(np.linalg.norm(p[I["Neck"]] - p[I["Hip"]])), 1e-6)
    heel = [(p[I[f"{side}Ankle"], 1] - p[I[f"{side}Heel"], 1]) / torso for side in ("L", "R")]
    torso_vec = p[I["Neck"]] - p[I["Hip"]]
    torso_inclination = np.degrees(np.arctan2(abs(torso_vec[0]), abs(torso_vec[1]) + 1e-8)) / 180.0
    return np.asarray([
        *knee, *hip, *heel, torso_inclination,
        abs(knee[0] - knee[1]), abs(hip[0] - hip[1]),
    ], dtype=float)


class RealtimePostureMatcher:
    """Phase-aware, 2D-only match used for display; never mutates production state."""

    def __init__(self, normal_2d_refs: list[dict], *, ema_alpha: float = REALTIME_EMA_ALPHA):
        self.normal_2d_refs = list(normal_2d_refs)
        self.ema_alpha = float(ema_alpha)
        self.phase_profiles = self._build_phase_profiles()
        self._ema: float | None = None
        self._phase: str | None = None
        self.timings_ms: list[float] = []

    @staticmethod
    def _phase_slices(ref: dict) -> dict[str, tuple[int, int]]:
        raw = np.asarray(ref["raw"])
        size = len(raw)
        bounds = ref.get("bounds") or {}
        translated = {}
        for state, label in PHASE_NAMES.items():
            if label in bounds:
                start, end = bounds[label]
                translated[state] = (max(0, int(start)), min(size, max(int(start) + 1, int(end))))
        if len(translated) == 4:
            return translated

        feat = np.asarray(ref["feat"])
        base_knee = np.median(feat[:min(5, size), :2].mean(axis=1))
        base_hip = np.median(feat[:min(5, size), 2:4].mean(axis=1))
        progress = np.maximum(0.0, base_knee - feat[:, :2].mean(axis=1)) + np.maximum(0.0, base_hip - feat[:, 2:4].mean(axis=1))
        bottom = int(np.argmax(progress))
        return {
            "prep": (0, min(size, max(2, min(5, bottom)))),
            "descend": (min(size - 1, 1), max(2, bottom)),
            "bottom": (max(0, bottom - 1), min(size, bottom + 2)),
            "ascend": (bottom, size),
        }

    def _build_phase_profiles(self) -> dict[str, dict]:
        profiles = {}
        for phase in PHASE_NAMES:
            rows = []
            for ref in self.normal_2d_refs:
                start, end = self._phase_slices(ref)[phase]
                rows.extend(_single_frame_2d_features(frame) for frame in np.asarray(ref["raw"])[start:end])
            if not rows:
                continue
            values = np.stack(rows)
            p10, median, p90 = np.percentile(values, (10, 50, 90), axis=0)
            width = np.maximum(p90 - p10, np.maximum(np.abs(median) * 0.05, 1e-4))
            profiles[phase] = {"p10": p10, "median": median, "p90": p90, "width": width, "samples": len(values)}
        return profiles

    def update(self, points, visibility, *, phase: str | None, baseline_ready: bool) -> dict:
        started = time.perf_counter()
        invalid_reason = None
        if not baseline_ready:
            invalid_reason = "baseline_not_ready"
        elif phase not in self.phase_profiles:
            invalid_reason = "phase_reference_unavailable"
        else:
            p = np.asarray(points, dtype=float)
            if p.shape != (18, 2) or not np.isfinite(p).all():
                invalid_reason = "invalid_landmarks"
            elif not visibility or any(float(visibility.get(name, 0.0)) < 0.4 for name in REQUIRED_VISIBILITY):
                invalid_reason = "insufficient_visibility"

        if invalid_reason is not None:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.timings_ms.append(elapsed)
            return {"match_valid": False, "match_percent": None, "raw_match": None,
                    "phase": phase, "invalid_reason": invalid_reason, "calculation_ms": elapsed}

        profile = self.phase_profiles[phase]
        current = _single_frame_2d_features(points)
        below = np.maximum(profile["p10"] - current, 0.0)
        above = np.maximum(current - profile["p90"], 0.0)
        excess = below + above
        feature_match = 100.0 * profile["width"] / (profile["width"] + excess)
        raw_match = float(np.clip(np.mean(feature_match), 0.0, 100.0))
        if phase != self._phase or self._ema is None:
            self._ema = raw_match
        else:
            self._ema = self.ema_alpha * raw_match + (1.0 - self.ema_alpha) * self._ema
        self._phase = phase
        elapsed = (time.perf_counter() - started) * 1000.0
        self.timings_ms.append(elapsed)
        self.timings_ms = self.timings_ms[-1800:]
        return {
            "match_valid": True, "match_percent": float(self._ema), "raw_match": raw_match,
            "phase": phase, "invalid_reason": None, "calculation_ms": elapsed,
            "reference_samples": int(profile["samples"]),
        }


__all__ = [
    "PostureScorer", "PostureScoreResult", "RealtimePostureMatcher",
    "COMPONENTS", "REALTIME_EMA_ALPHA",
]
