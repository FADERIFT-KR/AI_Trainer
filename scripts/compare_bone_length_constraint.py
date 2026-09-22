"""Create a diagnostic source-vs-length-constrained video/report from pose JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_trainer.bone_length_constraint import calibrate_leg_lengths, constrain_sequence
from ai_trainer.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS, COMMON_JOINT_NAMES
from ai_trainer.render import draw_skeleton_panel, fit_transform

IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
FOOT_TRIANGLE_PAIRS = COMMON_BONE_INDEX_PAIRS + [(IDX["LHeel"], IDX["LBigToe"]), (IDX["RHeel"], IDX["RBigToe"])]
FOOT_TRIANGLE_COLORS = COMMON_BONE_COLORS_BGR + [(255, 120, 0), (0, 60, 255)]
SEGMENTS = [("L thigh",6,8),("R thigh",7,9),("L shin",8,10),("R shin",9,11),
            ("L ankle-heel",10,12),("R ankle-heel",11,13),("L ankle-toe",10,14),("R ankle-toe",11,15),
            ("L heel-toe",12,14),("R heel-toe",13,15)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def lengths(sequence):
    return np.stack([np.linalg.norm(sequence[:, b] - sequence[:, a], axis=1) for _,a,b in SEGMENTS], axis=1)


def joint_angle(sequence, a, b, c):
    u, v = sequence[:, a] - sequence[:, b], sequence[:, c] - sequence[:, b]
    cosine = np.sum(u*v, axis=1) / (np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1))
    return np.degrees(np.arccos(np.clip(cosine,-1,1)))


def main():
    args = parse_args()
    records = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = np.asarray([record.get("common3d") for record in records], dtype=float)
    if source.shape != (len(records), 18, 3) or not np.isfinite(source).all():
        raise ValueError("Every JSONL row must contain finite common3d (18,3)")
    calibration = calibrate_leg_lengths(source)
    constrained = constrain_sequence(source, calibration)
    before, after = lengths(source), lengths(constrained)

    out_dir = args.out_dir or args.trace.parent / "bone_length_constraint"
    out_dir.mkdir(parents=True, exist_ok=False)
    projections = np.concatenate([source[:,:,[0,1]], source[:,:,[2,1]], constrained[:,:,[0,1]], constrained[:,:,[2,1]]])
    transform = fit_transform(projections, 480, 360, margin=30, flip_y=True)
    target = out_dir / "source_vs_constrained.mp4"
    timestamps = np.asarray([record.get("timestamp", np.nan) for record in records], dtype=float)
    video_fps = float(1.0 / np.median(np.diff(timestamps))) if len(timestamps)>1 and np.isfinite(timestamps).all() else 10.0
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (960, 780))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {target}")
    try:
        for index, (raw, fixed) in enumerate(zip(source, constrained)):
            canvas = np.zeros((780, 960, 3), dtype=np.uint8)
            record = records[index]
            cv2.putText(canvas, f"row={index} observation_id={record.get('observation_id')} | diagnostic only; not game input",
                        (12,24),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
            views = [(raw[:,[0,1]],"Source common3d XY"),(raw[:,[2,1]],"Source common3d ZY"),
                     (fixed[:,[0,1]],"Length-constrained XY"),(fixed[:,[2,1]],"Length-constrained ZY")]
            for panel,(points,title) in enumerate(views):
                x=(panel%2)*480;y=40+(panel//2)*360
                draw_skeleton_panel(canvas,(x,y),480,360,transform(points),title,
                                    "+Y up | fixed display scale",FOOT_TRIANGLE_PAIRS,FOOT_TRIANGLE_COLORS)
            cv2.putText(canvas,"Hip stays fixed; leg directions preserved; no ground/contact constraint",
                        (12,770),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1,cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()

    report = {
        "trace": str(args.trace.resolve()),
        "calibration_rows": list(calibration.source_indices),
        "units": "same as common3d (meters)",
        "connected_to_game": False,
        "constraints": "Hip fixed; observed thigh/shin directions retained; calibrated leg lengths and rigid foot triangle",
        "limitations": "No ground/contact constraint; no claim of improved ground-truth 3D accuracy",
        "segments": {},
        "video": str(target.resolve()),
        "video_fps_from_median_observation_interval": video_fps,
        "maximum_knee_angle_change_deg": {
            side: float(np.max(np.abs(joint_angle(source,*joints)-joint_angle(constrained,*joints))))
            for side,joints in (("left",(6,8,10)),("right",(7,9,11)))
        },
        "joint_correction_mm": {},
    }
    for name,index in (("LKnee",8),("RKnee",9),("LAnkle",10),("RAnkle",11),
                       ("LHeel",12),("RHeel",13),("LBigToe",14),("RBigToe",15)):
        displacement=np.linalg.norm(constrained[:,index]-source[:,index],axis=1)*1000.0
        report["joint_correction_mm"][name]={"median":float(np.median(displacement)),
                                              "p95":float(np.percentile(displacement,95)),
                                              "max":float(displacement.max())}
    for column,(name,_,_) in enumerate(SEGMENTS):
        base = float(np.median(before[list(calibration.source_indices), column]))
        report["segments"][name] = {
            "calibrated": base,
            "source_min": float(before[:,column].min()), "source_max": float(before[:,column].max()),
            "source_range_percent_of_calibration": float((before[:,column].max()-before[:,column].min())/base*100),
            "constrained_min": float(after[:,column].min()), "constrained_max": float(after[:,column].max()),
            "constrained_max_abs_error": float(np.max(np.abs(after[:,column]-base))),
        }
    (out_dir / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    main()
