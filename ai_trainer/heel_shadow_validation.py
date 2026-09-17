"""Lightweight webcam shadow logging for direct heel evidence.

This module is diagnostic-only.  It consumes a completed REP snapshot and
never returns or mutates a production classification.
"""
from __future__ import annotations

import csv
import json
import math
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

I = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
CLASSES = ("정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류")
FIELDS = [
    "session_id", "rep_id", "timestamp", "raw_dtw_class", "final_production_class",
    "unknown_reason", "distance_normal", "distance_heel", "distance_hip_down",
    "distance_hip_error", "production_heel_semantic_score", "production_heel_semantic_verdict",
    "heel_no_threshold", "heel_yes_threshold", "heel_toe_range_mean",
    "left_heel_toe_range", "right_heel_toe_range", "heel_vertical_max_abs_mean",
    "left_heel_vertical_max_abs", "right_heel_vertical_max_abs", "torso_length_px",
    "rep_frame_count", "feature_compute_ms", "log_write_ms", "ground_truth_label",
    "ground_truth_note",
]


def direct_heel_features(points: np.ndarray) -> dict[str, float | int]:
    """Match Patch-3 offline definitions on a common-skeleton 2-D REP.

    Image coordinates use +x right and +y down.  The scale and standing heel
    baseline are medians over the first five REP frames, exactly as in the
    offline pre-audit.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (18, 2) or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("completed REP points must be finite with shape (T,18,2), T >= 2")
    prep = min(5, len(p))
    torso = float(np.median(np.linalg.norm(
        p[:prep, I["Neck"]] - p[:prep, I["Hip"]], axis=1
    )))
    if not math.isfinite(torso) or torso <= 1e-6:
        raise ValueError("standing torso length is invalid")
    toe_ranges, heel_vertical = [], []
    result: dict[str, float | int] = {"torso_length_px": torso, "rep_frame_count": len(p)}
    for side, prefix in (("L", "left"), ("R", "right")):
        heel_y = p[:, I[side + "Heel"], 1]
        toe_y = p[:, I[side + "BigToe"], 1]
        heel_toe = (heel_y - toe_y) / torso
        heel_baseline = float(np.median(heel_y[:prep]))
        vertical = (heel_y - heel_baseline) / torso
        toe_range = float(np.ptp(heel_toe))
        vertical_max = float(np.max(np.abs(vertical)))
        result[prefix + "_heel_toe_range"] = toe_range
        result[prefix + "_heel_vertical_max_abs"] = vertical_max
        toe_ranges.append(toe_range); heel_vertical.append(vertical_max)
    result["heel_toe_range_mean"] = float(np.mean(toe_ranges))
    result["heel_vertical_max_abs_mean"] = float(np.mean(heel_vertical))
    return result


class HeelShadowValidationLogger:
    """One isolated, file-only logger per webcam session."""

    def __init__(self, root: str | Path, *, now: datetime | None = None):
        now = now or datetime.now()
        self.session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.path = Path(root) / "output/diagnostics/heel_shadow_validation" / self.session_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows: list[dict] = []
        self.failures: list[str] = []
        self._lock = threading.Lock()
        self.closed = False
        self._write_all()

    def safe_record(self, *, result, production: dict, raw2d: np.ndarray,
                    heel_thresholds: dict) -> dict | None:
        """Record best-effort and swallow every diagnostic failure."""
        try:
            return self.record(result=result, production=production, raw2d=raw2d,
                               heel_thresholds=heel_thresholds)
        except Exception as error:  # diagnostic isolation boundary
            message = f"{type(error).__name__}: {error}"
            self.failures.append(message)
            print(f"[HEEL-DIRECT-SHADOW WARNING] {message}", flush=True)
            return None

    def record(self, *, result, production: dict, raw2d: np.ndarray,
               heel_thresholds: dict) -> dict:
        started = time.perf_counter()
        feature = direct_heel_features(raw2d)
        compute_ms = (time.perf_counter() - started) * 1000.0
        debug = getattr(result, "debug_summary", None) or {}
        semantic = debug.get("heel_semantic") or {}
        distances = production.get("class_distances") or {}
        row = {
            "session_id": self.session_id, "rep_id": int(result.rep_index) + 1,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "raw_dtw_class": production.get("raw_3d_dtw"),
            "final_production_class": production.get("final"),
            "unknown_reason": production.get("reason_code"),
            "distance_normal": distances.get(CLASSES[0]), "distance_heel": distances.get(CLASSES[1]),
            "distance_hip_down": distances.get(CLASSES[2]), "distance_hip_error": distances.get(CLASSES[3]),
            "production_heel_semantic_score": semantic.get("value"),
            "production_heel_semantic_verdict": semantic.get("verdict"),
            "heel_no_threshold": heel_thresholds.get("no_evidence_max"),
            "heel_yes_threshold": heel_thresholds.get("positive_evidence_min"),
            **feature, "feature_compute_ms": compute_ms, "log_write_ms": 0.0,
            "ground_truth_label": "", "ground_truth_note": "",
        }
        with self._lock:
            expected = len(self.rows) + 1
            if row["rep_id"] != expected:
                raise ValueError(f"REP ordering mismatch: expected {expected}, got {row['rep_id']}")
            self.rows.append(row)
            write_started = time.perf_counter(); self._write_all()
            row["log_write_ms"] = (time.perf_counter() - write_started) * 1000.0
            self._write_all()
        print(
            f"[HEEL-DIRECT-SHADOW] rep={row['rep_id']} "
            f"toe_range={row['heel_toe_range_mean']:.5f} "
            f"heel_vertical={row['heel_vertical_max_abs_mean']:.5f}", flush=True,
        )
        return dict(row)

    def summary(self) -> dict:
        def stats(key):
            values = [float(r[key]) for r in self.rows]
            return {"count": len(values), "avg": float(np.mean(values)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None,
                    "max": max(values) if values else None}
        return {
            "session_id": self.session_id, "completed_reps": len(self.rows),
            "logger_failures": list(self.failures), "production_behavior_changed": False,
            "ground_truth_rows": sum(bool(r["ground_truth_label"]) for r in self.rows),
            "feature_compute_ms": stats("feature_compute_ms"), "log_write_ms": stats("log_write_ms"),
            "feature_definition": {
                "heel_toe_range_mean": "mean L/R ptp((heel_y-toe_y)/standing_torso)",
                "heel_vertical_max_abs_mean": "mean L/R max(abs((heel_y-median(first5 heel_y))/standing_torso))",
                "standing_torso": "median first-five-frame norm(Neck-Hip)",
            },
            "domain_mapping": {
                "live_heel": "MediaPipe LEFT/RIGHT_HEEL 29/30 -> common L/RHeel",
                "live_toe": "MediaPipe LEFT/RIGHT_FOOT_INDEX 31/32 -> common L/RBigToe",
                "offline_heel_toe": "AI Hub L/RHeel and L/RBigToe",
                "coordinate_orientation": "+x right, +y down in both image domains",
                "known_difference": "MediaPipe estimated/frozen landmarks and detector REP boundaries replace AI Hub annotations",
            },
            "ground_truth_policy": "ground_truth_label and note are intentionally blank",
        }

    def _write_csv(self, filename: str, rows: list[dict], fields=FIELDS):
        with (self.path / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)

    def _write_all(self):
        self._write_csv("reps.csv", self.rows)
        self._write_csv("annotation_template.csv", self.rows, [
            "session_id", "rep_id", "timestamp", "raw_dtw_class", "final_production_class",
            "production_heel_semantic_verdict", "heel_toe_range_mean",
            "heel_vertical_max_abs_mean", "ground_truth_label", "ground_truth_note",
        ])
        (self.path / "session_summary.json").write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

    def close(self):
        with self._lock:
            if not self.closed:
                self._write_all(); self.closed = True


__all__ = ["HeelShadowValidationLogger", "direct_heel_features", "FIELDS"]
