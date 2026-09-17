"""Completed-REP angle/domain diagnostics that never feed production decisions."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

I = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
SIDES = ("L", "R")
JOINTS = ("LShoulder", "RShoulder", "Neck", "Hip", "LHip", "RHip",
          "LKnee", "RKnee", "LAnkle", "RAnkle")


def interior_angle(a, b, c) -> float:
    u, v = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    return float(np.degrees(np.arccos(np.clip(
        np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8), -1.0, 1.0
    ))))


def angle_definition() -> dict:
    def item(names):
        return {"joints": list(names), "indices": [I[name] for name in names],
                "convention": "interior angle, 0..180 degrees"}
    return {
        "skeleton_order": list(COMMON_JOINT_NAMES),
        "hip": {
            "left": item(("Neck", "LHip", "LKnee")),
            "right": item(("Neck", "RHip", "RKnee")),
        },
        "knee": {
            "left": item(("LHip", "LKnee", "LAnkle")),
            "right": item(("RHip", "RKnee", "RAnkle")),
        },
        "standing_expectation": "angles near 160..180 degrees",
        "bottom_expectation": "angles decrease",
        "excursion": "standing angle minus minimum angle over the sequence",
    }


def frame_angles(frame) -> dict:
    p = np.asarray(frame, dtype=float)
    hip = [interior_angle(p[I["Neck"]], p[I[f"{s}Hip"]], p[I[f"{s}Knee"]]) for s in SIDES]
    knee = [interior_angle(p[I[f"{s}Hip"]], p[I[f"{s}Knee"]], p[I[f"{s}Ankle"]]) for s in SIDES]
    return {"hip_lr": hip, "knee_lr": knee,
            "hip_avg": float(np.mean(hip)), "knee_avg": float(np.mean(knee))}


def _summary(sequence, *, standing_frames=None, detector_bottom=None) -> dict:
    seq = np.asarray(sequence, dtype=float)
    angles = [frame_angles(frame) for frame in seq]
    if standing_frames is None:
        standing_angles = angles[:min(5, len(angles))]
        standing_indices = list(range(min(5, len(angles))))
    else:
        standing_angles = [frame_angles(frame) for frame in np.asarray(standing_frames)]
        standing_indices = None
    standing = {
        joint: {
            "left": float(np.median([row[f"{joint}_lr"][0] for row in standing_angles])),
            "right": float(np.median([row[f"{joint}_lr"][1] for row in standing_angles])),
        }
        for joint in ("hip", "knee")
    }
    for joint in standing:
        standing[joint]["avg"] = float((standing[joint]["left"] + standing[joint]["right"]) / 2.0)
    progress = np.asarray([
        max(0.0, standing["hip"]["avg"] - row["hip_avg"])
        + max(0.0, standing["knee"]["avg"] - row["knee_avg"])
        for row in angles
    ])
    bottom_index = int(np.argmax(progress))
    bottom = {joint: {
        "left": float(angles[bottom_index][f"{joint}_lr"][0]),
        "right": float(angles[bottom_index][f"{joint}_lr"][1]),
        "avg": float(angles[bottom_index][f"{joint}_avg"]),
    } for joint in ("hip", "knee")}
    excursion = {}
    minimum_indices = {}
    for joint in ("hip", "knee"):
        sides = []
        for side_index, side in enumerate(("left", "right")):
            values = np.asarray([row[f"{joint}_lr"][side_index] for row in angles])
            sides.append(float(standing[joint][side] - np.min(values)))
            minimum_indices[side + "_" + joint] = int(np.argmin(values))
        avg_values = np.asarray([row[f"{joint}_avg"] for row in angles])
        excursion[joint] = {
            "left": sides[0], "right": sides[1],
            "avg": float(standing[joint]["avg"] - np.min(avg_values)),
            "left_right_difference": float(abs(sides[0] - sides[1])),
        }
        minimum_indices["avg_" + joint] = int(np.argmin(avg_values))
    return {
        "standing": standing, "bottom": bottom, "excursion": excursion,
        "standing_indices": standing_indices, "bottom_index": bottom_index,
        "detector_bottom_index": detector_bottom,
        "detector_bottom_offset": None if detector_bottom is None else int(detector_bottom - bottom_index),
        "minimum_angle_indices": minimum_indices,
    }


def _normalized_landmarks(frame) -> dict:
    p = np.asarray(frame, dtype=float)
    scale = max(float(np.linalg.norm(p[I["Neck"]] - p[I["Hip"]])), 1e-8)
    normalized = (p - p[I["Hip"]]) / scale
    return {name: normalized[I[name]].tolist() for name in JOINTS}


def _triplet_stats(rows, path) -> dict:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return {"min": float(np.min(values)), "median": float(np.median(values)), "max": float(np.max(values))}


class AngleDomainDiagnostic:
    """Writes reference calibration once and one immutable row per completed REP."""

    def __init__(self, normal_2d_refs, normal_3d_refs, output_dir, semantic_cfg):
        self.normal_2d_refs = list(normal_2d_refs)
        self.normal_3d_refs = list(normal_3d_refs)
        self.semantic_cfg = dict(semantic_cfg)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = output / f"angle_domain_diagnostic_{stamp}.jsonl"
        self._stream = self.path.open("x", encoding="utf-8", buffering=1)
        self.reference_rows = self._reference_rows()
        self._write({"record_type": "angle_definition", **angle_definition()})
        self._write({"record_type": "normal_reference_summary",
                     "references": self.reference_rows,
                     "statistics": self._reference_statistics()})
        definition = angle_definition()
        print("\n[ANGLE DEFINITION]", flush=True)
        for joint in ("hip", "knee"):
            for side in ("left", "right"):
                row = definition[joint][side]
                print(f"Reference/Live 2D/3D {joint} {side}: {row['joints']} indices={row['indices']}", flush=True)
        print("definition same: YES", flush=True)

    def _reference_rows(self):
        rows = []
        count = min(len(self.normal_2d_refs), len(self.normal_3d_refs))
        for index in range(count):
            ref2d, ref3d = self.normal_2d_refs[index], self.normal_3d_refs[index]
            coords3d = np.asarray(ref3d["feat"]["joint_coords_3d"]).reshape(-1, 18, 3)
            summary2d = _summary(ref2d["raw"])
            summary3d = _summary(coords3d)
            rows.append({
                "reference_id_2d": ref2d.get("id", f"normal_2d_{index}"),
                "reference_id_3d": ref3d.get("meta", {}).get("medoid_id", f"normal_3d_{index}"),
                "2d": summary2d, "3d": summary3d,
                "existing_scorer_2d": {
                    "hip_excursion": float(ref2d["summary"]["hip_excursion"]),
                    "knee_excursion": float(ref2d["summary"]["knee_excursion"]),
                },
                "value_consistency": {
                    "hip_difference": float(ref2d["summary"]["hip_excursion"] - summary2d["excursion"]["hip"]["avg"]),
                    "knee_difference": float(ref2d["summary"]["knee_excursion"] - summary2d["excursion"]["knee"]["avg"]),
                },
            })
        return rows

    def _reference_statistics(self):
        output = {}
        for domain in ("2d", "3d"):
            output[domain] = {}
            for joint in ("hip", "knee"):
                output[domain][joint] = {
                    "standing": _triplet_stats(self.reference_rows, (domain, "standing", joint, "avg")),
                    "bottom": _triplet_stats(self.reference_rows, (domain, "bottom", joint, "avg")),
                    "excursion": _triplet_stats(self.reference_rows, (domain, "excursion", joint, "avg")),
                }
        return output

    def record(self, *, rep, raw2d, coords3d, standing2d_frames, standing3d_frames,
               detector_bottom, scorer_payload, production):
        summary2d = _summary(raw2d, standing_frames=standing2d_frames, detector_bottom=detector_bottom)
        summary3d = _summary(coords3d, standing_frames=standing3d_frames, detector_bottom=detector_bottom)
        scorer_diag = scorer_payload.get("diagnostic") or {}
        consistency = {}
        for joint in ("hip", "knee"):
            delta = summary3d["excursion"][joint]["avg"] - summary2d["excursion"][joint]["avg"]
            lo = float(self.semantic_cfg[f"{joint}_3d_minus_2d_excursion_min_deg"])
            hi = float(self.semantic_cfg[f"{joint}_3d_minus_2d_excursion_max_deg"])
            consistency[joint] = {"delta_3d_minus_2d": float(delta), "allowed": [lo, hi], "pass": lo <= delta <= hi}
        row = {
            "record_type": "completed_rep", "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "rep": int(rep), "production": dict(production), "live_2d": summary2d, "live_3d": summary3d,
            "value_consistency": {
                "existing_hip_excursion": scorer_diag.get("hip_excursion_2d"),
                "recomputed_hip_excursion": summary2d["excursion"]["hip"]["avg"],
                "existing_knee_excursion": scorer_diag.get("knee_excursion_2d"),
                "recomputed_knee_excursion": summary2d["excursion"]["knee"]["avg"],
            },
            "consistency_2d_3d": consistency,
            "normalized_landmarks": {
                "standing": _normalized_landmarks(np.asarray(standing2d_frames)[0]),
                "bottom": _normalized_landmarks(np.asarray(raw2d)[summary2d["bottom_index"]]),
            },
            "posture_match": scorer_payload,
        }
        for joint in ("hip", "knee"):
            existing = row["value_consistency"][f"existing_{joint}_excursion"]
            recomputed = row["value_consistency"][f"recomputed_{joint}_excursion"]
            row["value_consistency"][f"{joint}_difference"] = None if existing is None else float(existing - recomputed)
        self._write(row)
        print(f"\n[REP ANGLE DIAGNOSTIC] REP {rep}", flush=True)
        for domain, values in (("Live 2D", summary2d), ("Lifting 3D", summary3d)):
            print(f"{domain} hip standing/bottom/excursion: "
                  f"{values['standing']['hip']['avg']:.2f} / {values['bottom']['hip']['avg']:.2f} / {values['excursion']['hip']['avg']:.2f}", flush=True)
            print(f"{domain} knee standing/bottom/excursion: "
                  f"{values['standing']['knee']['avg']:.2f} / {values['bottom']['knee']['avg']:.2f} / {values['excursion']['knee']['avg']:.2f}", flush=True)
        print(f"2D→3D hip delta/pass: {consistency['hip']['delta_3d_minus_2d']:.2f} / {consistency['hip']['pass']}", flush=True)
        print(f"2D→3D knee delta/pass: {consistency['knee']['delta_3d_minus_2d']:.2f} / {consistency['knee']['pass']}", flush=True)

    def _write(self, row):
        self._stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None


__all__ = ["AngleDomainDiagnostic", "angle_definition", "frame_angles", "interior_angle"]
