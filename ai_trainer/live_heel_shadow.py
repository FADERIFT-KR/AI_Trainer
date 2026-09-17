"""File-only live heel-policy shadow logging; never changes production results."""
from __future__ import annotations

import csv
import json
import math
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from .heel_policy_shadow import HEEL, NORMAL, UNKNOWN, apply_policy, relative_heel_margin

SCHEMA = [
    "session_id", "rep_index", "timestamp", "raw_class", "raw_normal_distance",
    "raw_heel_distance", "raw_margin", "production_final", "production_reason",
    "heel_evidence", "heel_score", "heel_no_threshold", "heel_yes_threshold",
    "heel_2d_normal_distance", "heel_2d_error_distance", "heel_2d_prediction",
    "heel_2d_margin", "shadow_b_final", "shadow_b_reason", "shadow_d_final",
    "shadow_d_reason", "production_vs_d_disagree", "b_vs_d_disagree",
    "depth_2d_pass", "consistency_3d_pass", "completed_rep_valid",
    "heel_relative_motion", "left_heel_ankle_motion", "right_heel_ankle_motion",
    "left_heel_toe_delta", "right_heel_toe_delta", "left_heel_raw_json",
    "right_heel_raw_json", "shadow_compute_ms", "log_write_ms",
    "ground_truth_label", "ground_truth_note",
]
SUPPORTED_GROUND_TRUTH = (NORMAL, HEEL, "고관절오류", "엉덩이하방오류", "UNKNOWN", "UNCERTAIN")


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _two_class_gate(two_d: dict | None) -> tuple[float | None, float | None, str | None, float | None]:
    distances = (two_d or {}).get("distance_by_class") or {}
    normal, heel = _finite(distances.get(NORMAL)), _finite(distances.get(HEEL))
    if normal is None or heel is None:
        return normal, heel, None, None
    prediction = HEEL if heel < normal else NORMAL
    return normal, heel, prediction, relative_heel_margin(normal, heel)


class LiveHeelShadowLogger:
    """One logger per webcam session. All ground-truth fields stay blank."""
    def __init__(self, root: str | Path, *, now: datetime | None = None):
        now = now or datetime.now()
        self.session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.path = Path(root) / "output/diagnostics/live_heel_shadow" / self.session_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows: list[dict] = []
        self._lock = threading.Lock()
        self.closed = False
        self._write_all()

    def record(self, *, result, production: dict, two_d: dict | None,
               heel_thresholds: dict, two_d_compute_ms: float = 0.0) -> dict:
        started = time.perf_counter()
        debug = getattr(result, "debug_summary", None) or {}
        evidence = debug.get("heel_semantic") or {}
        semantic = (two_d or {}).get("semantic") or {}
        raw = production.get("raw_3d_dtw")
        prod_final = production.get("final")
        pre_heel_final = production.get("pre_heel_final", prod_final)
        normal3d = _finite((production.get("class_distances") or {}).get(NORMAL))
        heel3d = _finite((production.get("class_distances") or {}).get(HEEL))
        raw_margin = relative_heel_margin(normal3d, heel3d) if normal3d is not None and heel3d is not None else None
        normal2d, heel2d, direct, direct_margin = _two_class_gate(two_d)
        verdict = evidence.get("verdict", "AMBIGUOUS")
        if raw == HEEL and pre_heel_final == HEEL:
            shadow_b, reason_b = apply_policy("B_NO_REJECT", raw, verdict)
            shadow_d, reason_d = apply_policy("D_HEEL_DIRECT_2CLASS", raw, verdict, direct)
        else:
            shadow_b, reason_b = pre_heel_final, production.get("reason_code")
            shadow_d, reason_d = pre_heel_final, production.get("reason_code")
        compute_ms = (time.perf_counter() - started) * 1000.0 + max(0.0, float(two_d_compute_ms))
        row = {
            "session_id": self.session_id, "rep_index": int(result.rep_index) + 1,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"), "raw_class": raw,
            "raw_normal_distance": normal3d, "raw_heel_distance": heel3d, "raw_margin": raw_margin,
            "production_final": prod_final, "production_reason": production.get("reason_code"),
            "heel_evidence": verdict, "heel_score": _finite(evidence.get("value")),
            "heel_no_threshold": _finite(heel_thresholds.get("no_evidence_max")),
            "heel_yes_threshold": _finite(heel_thresholds.get("positive_evidence_min")),
            "heel_2d_normal_distance": normal2d, "heel_2d_error_distance": heel2d,
            "heel_2d_prediction": direct, "heel_2d_margin": direct_margin,
            "shadow_b_final": shadow_b, "shadow_b_reason": reason_b,
            "shadow_d_final": shadow_d, "shadow_d_reason": reason_d,
            "production_vs_d_disagree": prod_final != shadow_d, "b_vs_d_disagree": shadow_b != shadow_d,
            "depth_2d_pass": bool(production.get("depth_2d_pass")),
            "consistency_3d_pass": bool(production.get("consistency_pass")),
            "completed_rep_valid": True,
            "heel_relative_motion": _finite(semantic.get("heel_relative_motion")),
            "left_heel_ankle_motion": _finite(semantic.get("left_heel_ankle_motion")),
            "right_heel_ankle_motion": _finite(semantic.get("right_heel_ankle_motion")),
            "left_heel_toe_delta": _finite(semantic.get("left_heel_toe_delta")),
            "right_heel_toe_delta": _finite(semantic.get("right_heel_toe_delta")),
            "left_heel_raw_json": json.dumps(evidence.get("left") or {}, ensure_ascii=False),
            "right_heel_raw_json": json.dumps(evidence.get("right") or {}, ensure_ascii=False),
            "shadow_compute_ms": compute_ms, "log_write_ms": 0.0,
            "ground_truth_label": "", "ground_truth_note": "",
        }
        with self._lock:
            self.rows.append(row)
            write_started = time.perf_counter(); self._write_all()
            row["log_write_ms"] = (time.perf_counter() - write_started) * 1000.0
            self._write_all()
        print(f"[HEEL-SHADOW] rep={row['rep_index']} raw={raw} prod={prod_final} evidence={verdict} 2d_gate={direct or 'N/A'} shadowD={shadow_d}", flush=True)
        return dict(row)

    def summary(self) -> dict:
        evidence = Counter(row["heel_evidence"] for row in self.rows)
        timings = [row["shadow_compute_ms"] for row in self.rows]
        writes = [row["log_write_ms"] for row in self.rows]
        stats = lambda x: {"count": len(x), "avg": sum(x) / len(x) if x else None,
                           "p95": float(__import__("numpy").percentile(x, 95)) if x else None,
                           "max": max(x) if x else None}
        return {
            "session_id": self.session_id, "total_reps": len(self.rows),
            "raw_heel_candidates": sum(r["raw_class"] == HEEL for r in self.rows),
            "production_heel_finals": sum(r["production_final"] == HEEL for r in self.rows),
            "policy_b_heel_finals": sum(r["shadow_b_final"] == HEEL for r in self.rows),
            "policy_d_heel_finals": sum(r["shadow_d_final"] == HEEL for r in self.rows),
            "production_vs_d_disagreement_count": sum(r["production_vs_d_disagree"] for r in self.rows),
            "evidence_counts": {k: evidence.get(k, 0) for k in ("NO", "AMBIGUOUS", "YES")},
            "shadow_compute_ms": stats(timings), "log_write_ms": stats(writes),
            "ground_truth_rows": sum(bool(r["ground_truth_label"]) for r in self.rows),
            "ground_truth_note": "Predictions are never copied into ground truth fields.",
            "recommended_collection": {"completed_reps": "20+", "raw_heel_candidates": "10+",
                                       "normal_intent": "10+", "intentional_heel_lift": "10+"},
            "production_behavior_changed": False,
        }

    def _write_csv(self, name: str, rows: list[dict], fields=SCHEMA):
        with (self.path / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)

    def _write_all(self):
        self._write_csv("reps.csv", self.rows)
        self._write_csv("raw_heel_candidates.csv", [r for r in self.rows if r["raw_class"] == HEEL])
        self._write_csv("shadow_disagreements.csv", [r for r in self.rows if r["production_vs_d_disagree"] or r["b_vs_d_disagree"]])
        self._write_csv("annotation_template.csv", self.rows,
                        ["session_id", "rep_index", "timestamp", "raw_class", "production_final",
                         "heel_evidence", "heel_2d_prediction", "shadow_d_final",
                         "ground_truth_label", "ground_truth_note"])
        (self.path / "rep_details.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in self.rows), encoding="utf-8")
        (self.path / "session_summary.json").write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def close(self):
        with self._lock:
            if not self.closed:
                self._write_all(); self.closed = True


__all__ = ["LiveHeelShadowLogger", "SCHEMA", "SUPPORTED_GROUND_TRUTH", "_two_class_gate"]
