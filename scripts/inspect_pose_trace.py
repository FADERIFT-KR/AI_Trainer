"""Summarize a game pose trace without camera access."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.features import _angle_deg, extract_all_features
from ai_trainer.game_ui.pose_bridge import _DIRECT_MP_IDX
from ai_trainer.reference_db_io import load_reference_db


def raw_common(world):
    out = np.empty((len(world), 18, 3))
    for i, name in enumerate(COMMON_JOINT_NAMES):
        if name == "Hip":
            out[:, i] = world[:, [23, 24], :3].mean(axis=1)
        elif name == "Neck":
            out[:, i] = world[:, [11, 12], :3].mean(axis=1)
        else:
            out[:, i] = world[:, _DIRECT_MP_IDX[name], :3]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines()]
    if len(records) < 2:
        raise SystemExit("At least two recorded observations are required")
    world = np.array([r["world_landmarks"] for r in records])
    common = np.array([r["common3d"] for r in records])
    image = np.array([r["image_landmarks"] for r in records])[:, :, :2]
    image *= np.array([r["image_size"] for r in records])[:, None, :]
    raw = raw_common(world)
    idx = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
    raw_features = extract_all_features(raw)
    filtered_features = extract_all_features(common)
    times = np.array([r["timestamp"] for r in records])
    report = {"frames": len(records), "median_fps": float(1 / np.median(np.diff(times))), "reps": []}
    sample_indices = np.array([r.get("sample_index") if r.get("sample_index") is not None else -1 for r in records])
    for record_index, r in enumerate(records):
        rep = r.get("completed_rep")
        if not rep:
            continue
        s, e = rep["frame_range"]
        if r.get("schema_version", 1) >= 2:
            selected = np.flatnonzero((sample_indices >= s) & (sample_indices <= e))
            if selected.size == 0:
                continue
            s, e = int(selected[0]), min(record_index, int(selected[-1]) + 1)
        j = s + int(np.argmin(filtered_features["knee_flexion_angle"][s:e + 1].mean(axis=1)))
        bone_ratios = {}
        for side in ("L", "R"):
            for a, b in (("Hip", "Knee"), ("Knee", "Ankle")):
                lengths = np.linalg.norm(raw[:, idx[side+a]] - raw[:, idx[side+b]], axis=1)
                bone_ratios[side+a+b] = float(lengths[j] / np.median(lengths[:8]))
        report["reps"].append({
            "rep": rep["rep_index"], "class": rep["predicted_class"], "bottom_frame": j,
            "raw_knee_deg": (raw_features["knee_flexion_angle"][j] * 180).tolist(),
            "filtered_knee_deg": (filtered_features["knee_flexion_angle"][j] * 180).tolist(),
            "raw_hip_deg": (raw_features["hip_flexion_angle"][j] * 180).tolist(),
            "image_knee_deg": [float(_angle_deg(image[j,a],image[j,b],image[j,c])) for a,b,c in [(23,25,27),(24,26,28)]],
            "bone_length_ratio_to_start": bone_ratios,
            "max_knee_filter_delta_deg": float(np.max(np.abs(raw_features["knee_flexion_angle"][s:e+1] - filtered_features["knee_flexion_angle"][s:e+1]))*180),
            "phase_lengths": {p: b-a for p,(a,b) in rep["phase_bounds"].items()},
            "assessment_supported": rep.get("assessment_supported", True),
        })
    db = load_reference_db(Path(__file__).resolve().parents[1] / "output/reference_db")["ground_truth"]
    report["references"] = {c: [{"frames": len(m["feat"]["knee_flexion_angle"]), "minimum_knee_deg": (m["feat"]["knee_flexion_angle"].min(axis=0)*180).tolist(), "minimum_hip_deg": (m["feat"]["hip_flexion_angle"].min(axis=0)*180).tolist(), "maximum_torso_deg": float(m["feat"]["torso_inclination"].max()*180)} for m in refs] for c, refs in db.items()}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text+"\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
