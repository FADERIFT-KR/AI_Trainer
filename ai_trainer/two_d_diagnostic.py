"""Diagnostic-only 2-D pose comparison; never feeds production decisions."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .aihub_zip import AiHubZip
from .common_skeleton import COMMON_JOINT_NAMES, to_common_skeleton
from .dtw_compare import _dtw_dp
from .adapter_diagnostic import geometry_ratios

I = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def _angle(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8), -1, 1))))


def extract_2d_features(
    points: np.ndarray, *, baseline_knee: float | None = None,
    baseline_hip: float | None = None, bottom_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Return a scale-normalized temporal feature matrix and semantic summary."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (18, 2):
        raise ValueError("2D sequence must have shape (T,18,2)")
    if not np.isfinite(p).all() or len(p) < 2:
        raise ValueError("2D sequence is empty or non-finite")
    names = ("L", "R")
    knee = np.array([[_angle(f[I[s+"Hip"]], f[I[s+"Knee"]], f[I[s+"Ankle"]]) for s in names] for f in p])
    hip = np.array([[_angle(f[I["Neck"]], f[I[s+"Hip"]], f[I[s+"Knee"]]) for s in names] for f in p])
    ankle = np.array([[_angle(f[I[s+"Knee"]], f[I[s+"Ankle"]], f[I[s+"BigToe"]]) for s in names] for f in p])
    torso_len = np.linalg.norm(p[:, I["Neck"]] - p[:, I["Hip"]], axis=1)
    scale = max(float(np.median(torso_len[:min(5, len(p))])), 1e-6)
    base_hip_y = float(np.median(p[:min(5, len(p)), I["Hip"], 1]))
    pelvis = (p[:, I["Hip"], 1] - base_hip_y) / scale
    heel = np.stack([(p[:, I[s+"Ankle"], 1] - p[:, I[s+"Heel"], 1]) / scale for s in names], axis=1)
    heel_toe = np.stack([(p[:, I[s+"Heel"], 1] - p[:, I[s+"BigToe"], 1]) / scale for s in names], axis=1)
    root_heel = np.stack([(p[:, I[s+"Heel"], 1] - p[:, I["Hip"], 1]) / scale for s in names], axis=1)
    torso_vec = p[:, I["Neck"]] - p[:, I["Hip"]]
    torso_incl = np.degrees(np.arctan2(np.abs(torso_vec[:, 0]), np.abs(torso_vec[:, 1]) + 1e-8))
    feat = np.column_stack([knee/180.0, hip/180.0, ankle/180.0, pelvis, heel,
                            torso_incl/180.0, np.abs(knee[:, 0]-knee[:, 1])/180.0,
                            np.abs(hip[:, 0]-hip[:, 1])/180.0])
    base_knee = float(np.median(knee[:min(5,len(p))].mean(axis=1))) if baseline_knee is None else float(baseline_knee)
    base_hip = float(np.median(hip[:min(5,len(p))].mean(axis=1))) if baseline_hip is None else float(baseline_hip)
    progress = np.maximum(0, base_knee-knee.mean(axis=1)) + np.maximum(0, base_hip-hip.mean(axis=1))
    bottom = int(np.argmax(progress))
    phase_bottom = bottom
    if bottom_range is not None:
        lo=max(0,min(int(bottom_range[0]),len(p)-1));hi=max(lo+1,min(int(bottom_range[1]),len(p)))
        phase_bottom=lo+int(np.argmax(progress[lo:hi]))
    summary = {
        "bottom_frame_global": bottom,
        "bottom_knee_angle": float(knee[bottom].mean()),
        "bottom_hip_angle": float(hip[bottom].mean()),
        "bottom_ankle_angle": float(ankle[bottom].mean()),
        "bottom_frame_phase": phase_bottom,
        "bottom_frame_difference": int(phase_bottom-bottom),
        "phase_bottom_knee_angle": float(knee[phase_bottom].mean()),
        "phase_bottom_hip_angle": float(hip[phase_bottom].mean()),
        "knee_excursion": float(base_knee - knee[:, :].mean(axis=1).min()),
        "hip_excursion": float(base_hip - hip[:, :].mean(axis=1).min()),
        "pelvis_vertical_excursion": float(np.max(pelvis) - np.min(pelvis)),
        "heel_relative_motion": float(np.max(heel) - np.min(heel)),
        "left_heel_ankle_motion": float(np.ptp(heel[:, 0])),
        "right_heel_ankle_motion": float(np.ptp(heel[:, 1])),
        "left_heel_ankle_standing": float(np.median(heel[:min(5, len(p)), 0])),
        "right_heel_ankle_standing": float(np.median(heel[:min(5, len(p)), 1])),
        "left_heel_ankle_bottom": float(heel[bottom, 0]),
        "right_heel_ankle_bottom": float(heel[bottom, 1]),
        "left_heel_toe_delta": float(heel_toe[bottom, 0] - np.median(heel_toe[:min(5, len(p)), 0])),
        "right_heel_toe_delta": float(heel_toe[bottom, 1] - np.median(heel_toe[:min(5, len(p)), 1])),
        "left_root_centered_heel_delta": float(root_heel[bottom, 0] - np.median(root_heel[:min(5, len(p)), 0])),
        "right_root_centered_heel_delta": float(root_heel[bottom, 1] - np.median(root_heel[:min(5, len(p)), 1])),
        "torso_inclination_bottom": float(torso_incl[bottom]),
        "left_right_knee_asymmetry_bottom": float(abs(knee[bottom,0]-knee[bottom,1])),
        "left_right_hip_asymmetry_bottom": float(abs(hip[bottom,0]-hip[bottom,1])),
    }
    return feat.astype(np.float32), summary


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    total, path = _dtw_dp(cost, window_ratio=None)
    return float(total / max(path, 1))


class TwoDDiagnostic:
    def __init__(self, root: str | Path, tl_zip: str | Path, vl_zip: str | Path):
        self.root = Path(root)
        self.enabled = True
        self.error = None
        self.refs: dict[str, list[dict]] = {}
        self.output_dir = self.root / "output" / "diagnostics"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = self.output_dir / f"2d_only_{stamp}.jsonl"
        self.report_path = self.output_dir / f"2d_reference_distribution_{stamp}.json"
        self._stream = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("x", encoding="utf-8", buffering=1)
            self._load_refs(tl_zip, vl_zip)
            report = self.reference_report()
            self.report = report
            self.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write({"record_type": "reference_report", "report_file": self.report_path.name, **report})
        except Exception as error:
            self.enabled = False; self.error = f"{type(error).__name__}: {error}"

    def _load_refs(self, tl_zip, vl_zip):
        manifest = json.loads((self.root/"output"/"reference_db"/"manifest.json").read_text(encoding="utf-8"))
        zips = {"TL": AiHubZip(tl_zip), "VL": AiHubZip(vl_zip)}
        try:
            for entry in manifest["entries"]:
                if entry["tier"] != "operational": continue
                z=zips[entry["origin_zip"]]
                candidates=z.find_sequences(error_type=entry["class_label"], level=entry["difficulty_level"], actor=entry["actor_id"], rep=entry["repetition_id"])
                if len(candidates)!=1: raise ValueError(f"reference source match count={len(candidates)}: {entry['medoid_id']}")
                _,p=z.read_2d(candidates[0],1); start,end=entry["frame_range"]
                raw=to_common_skeleton(p[start:end+1]); feat,summary=extract_2d_features(raw)
                self.refs.setdefault(entry["class_label"],[]).append({"id":entry["medoid_id"],"raw":raw,"feat":feat,"summary":summary,"bounds":entry["phase_boundaries"]})
        finally:
            for z in zips.values(): z.close()

    def reference_report(self):
        fields=list(next(iter(next(iter(self.refs.values()))))["summary"].keys())
        fields=[x for x in fields if not x.startswith("bottom_frame")]
        dist={}
        for cls,refs in self.refs.items():
            dist[cls]={}
            for field in fields:
                a=np.array([r["summary"][field] for r in refs]);dist[cls][field]={k:float(v) for k,v in zip(("min","p10","median","p90","max"),np.percentile(a,[0,10,50,90,100]))}
        normal=self.refs.get("정상",[]); overlaps={}
        for cls,refs in self.refs.items():
            if cls=="정상":continue
            overlaps[cls]={}
            for field in fields:
                a=np.array([r["summary"][field] for r in normal]);b=np.array([r["summary"][field] for r in refs]);lo=max(np.percentile(a,10),np.percentile(b,10));hi=min(np.percentile(a,90),np.percentile(b,90));union=max(np.percentile(a,90),np.percentile(b,90))-min(np.percentile(a,10),np.percentile(b,10));overlaps[cls][field]=float(max(0,hi-lo)/max(union,1e-8))
        geometry = {}
        for cls, refs in self.refs.items():
            geometry[cls] = {}
            for phase in ("prep", "bottom"):
                rows = []
                for ref in refs:
                    raw = ref["raw"]
                    if phase == "prep":
                        frames = raw[:min(5, len(raw))]
                    else:
                        idx = int(ref["summary"]["bottom_frame_global"])
                        frames = raw[idx:idx + 1]
                    rows.append({k: float(np.median([geometry_ratios(f)[k] for f in frames]))
                                 for k in geometry_ratios(frames[0])})
                geometry[cls][phase] = {
                    field: {k: float(v) for k, v in zip(("min", "p10", "median", "p90", "max"),
                             np.percentile([row[field] for row in rows], [0, 10, 50, 90, 100]))}
                    for field in rows[0]
                }
        return {"classes":list(self.refs),"reference_count":{k:len(v) for k,v in self.refs.items()},"distributions":dist,"normal_overlap_p10_p90":overlaps,"geometry_distributions":geometry}

    def diagnose(
        self, raw: np.ndarray, production: dict, *, baseline_knee: float | None = None,
        baseline_hip: float | None = None, bottom_range: tuple[int, int] | None = None,
        hip_geometry: dict | None = None,
    ) -> dict:
        try:
            feat,semantic=extract_2d_features(raw, baseline_knee=baseline_knee,
                                             baseline_hip=baseline_hip, bottom_range=bottom_range); distances={};ids={}
            for cls,refs in self.refs.items():
                scored=[(_distance(feat,r["feat"]),r["id"]) for r in refs];distances[cls],ids[cls]=min(scored)
            ordered=sorted(distances,key=distances.get);margin=float(distances[ordered[1]]-distances[ordered[0]])
            relative=margin/max(float(distances[ordered[0]]),1e-8)
            # Diagnostic confidence only; it never changes production. Five percent
            # relative separation is reported explicitly rather than hidden.
            best=ordered[0] if relative>=0.05 else "2D_ONLY_UNKNOWN"
            out={"record_type":"completed_rep","timestamp":datetime.now().isoformat(timespec="milliseconds"),"production":production,"distance_by_class":distances,"best_reference_by_class":ids,"raw_best":ordered[0],"best":best,"margin":margin,"relative_margin":relative,"margin_rule":">= 0.05 diagnostic-only","semantic":semantic,"hip_geometry":hip_geometry}
            self._write(out);return out
        except Exception as error:
            return {"error":f"{type(error).__name__}: {error}","best":"2D_DIAGNOSTIC_FAILED"}

    def _write(self,row):
        if self.enabled and self._stream:
            try:self._stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            except Exception as error:self.error=f"{type(error).__name__}: {error}";self.enabled=False

    def close(self):
        try:
            if self._stream:self._stream.close()
        except Exception:pass


__all__=["TwoDDiagnostic","extract_2d_features"]
