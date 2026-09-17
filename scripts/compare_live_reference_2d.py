"""Read-only comparison of normal AI Hub 2D bottoms and two captured live bottoms."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trainer.aihub_zip import AiHubZip
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES, to_common_skeleton
from ai_trainer.lifting_parity import joint_angle
from ai_trainer.normalization import normalize_2d_sequence
from ai_trainer.reference_db_io import load_reference_db
from scripts.build_reference_db import TL_ZIP

IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}

LIVE = {
    "REP 1": {
        "LHip": [674.3910217285156, 490.67087173461914],
        "RHip": [621.153678894043, 491.55462741851807],
        "LKnee": [712.2777557373047, 551.7050743103027],
        "RKnee": [592.8553009033203, 552.0236349105835],
        "LAnkle": [704.0716552734375, 628.6903524398804],
        "RAnkle": [604.4136428833008, 628.743052482605],
    },
    "REP 2": {
        "LHip": [673.7189483642578, 503.4304618835449],
        "RHip": [622.3055267333984, 503.65426540374756],
        "LKnee": [713.7510681152344, 555.7320356369019],
        "RKnee": [595.0022506713867, 550.1969003677368],
        "LAnkle": [705.1972961425781, 628.9742374420166],
        "RAnkle": [603.5258865356445, 628.5578727722168],
    },
}


def distance(frame, a, b):
    return float(np.linalg.norm(frame[IDX[a]] - frame[IDX[b]]))


def metrics(frame):
    hip_l = joint_angle(frame[IDX["Neck"]], frame[IDX["LHip"]], frame[IDX["LKnee"]])
    hip_r = joint_angle(frame[IDX["Neck"]], frame[IDX["RHip"]], frame[IDX["RKnee"]])
    knee_l = joint_angle(frame[IDX["LHip"]], frame[IDX["LKnee"]], frame[IDX["LAnkle"]])
    knee_r = joint_angle(frame[IDX["RHip"]], frame[IDX["RKnee"]], frame[IDX["RAnkle"]])
    ankle_l = joint_angle(frame[IDX["LKnee"]], frame[IDX["LAnkle"]], frame[IDX["LBigToe"]])
    ankle_r = joint_angle(frame[IDX["RKnee"]], frame[IDX["RAnkle"]], frame[IDX["RBigToe"]])
    neck_hip = distance(frame, "Neck", "Hip")
    hip_width = distance(frame, "LHip", "RHip")
    thigh_l = distance(frame, "LHip", "LKnee")
    thigh_r = distance(frame, "RHip", "RKnee")
    shin_l = distance(frame, "LKnee", "LAnkle")
    shin_r = distance(frame, "RKnee", "RAnkle")
    ankle_distance = distance(frame, "LAnkle", "RAnkle")
    knee_distance = distance(frame, "LKnee", "RKnee")
    ankle_y = (frame[IDX["LAnkle"], 1] + frame[IDX["RAnkle"], 1]) / 2.0
    return {
        "hip_l": hip_l, "hip_r": hip_r, "hip_avg": (hip_l + hip_r) / 2.0,
        "knee_l": knee_l, "knee_r": knee_r, "knee_avg": (knee_l + knee_r) / 2.0,
        "ankle_l": ankle_l, "ankle_r": ankle_r, "ankle_avg": (ankle_l + ankle_r) / 2.0,
        "pelvis_height_over_neck_hip": float((ankle_y - frame[IDX["Hip"], 1]) / neck_hip),
        "hip_width_over_neck_hip": hip_width / neck_hip,
        "thigh_l_over_neck_hip": thigh_l / neck_hip,
        "thigh_r_over_neck_hip": thigh_r / neck_hip,
        "shin_l_over_neck_hip": shin_l / neck_hip,
        "shin_r_over_neck_hip": shin_r / neck_hip,
        "ankle_distance_over_hip_width": ankle_distance / hip_width,
        "knee_distance_over_hip_width": knee_distance / hip_width,
    }


def live_metrics(points):
    p = {name: np.asarray(value, dtype=np.float64) for name, value in points.items()}
    hip_width = float(np.linalg.norm(p["LHip"] - p["RHip"]))
    return {
        "hip_width_px": hip_width,
        "thigh_l_px": float(np.linalg.norm(p["LHip"] - p["LKnee"])),
        "thigh_r_px": float(np.linalg.norm(p["RHip"] - p["RKnee"])),
        "shin_l_px": float(np.linalg.norm(p["LKnee"] - p["LAnkle"])),
        "shin_r_px": float(np.linalg.norm(p["RKnee"] - p["RAnkle"])),
        "ankle_distance_over_hip_width": float(np.linalg.norm(p["LAnkle"] - p["RAnkle"])) / hip_width,
        "knee_distance_over_hip_width": float(np.linalg.norm(p["LKnee"] - p["RKnee"])) / hip_width,
    }


def summary(rows):
    keys = rows[0].keys()
    return {
        key: {
            "min": float(np.min([row[key] for row in rows])),
            "p10": float(np.percentile([row[key] for row in rows], 10)),
            "median": float(np.median([row[key] for row in rows])),
            "p90": float(np.percentile([row[key] for row in rows], 90)),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in keys
    }


def main():
    db = load_reference_db(ROOT / "output" / "reference_db")
    medoids = db["operational"]["정상"]
    rows = []
    details = []
    with AiHubZip(TL_ZIP) as archive:
        for medoid in medoids:
            meta = medoid["meta"]
            seq = archive.find_sequences(
                error_type="정상", actor=meta["actor_id"], rep=meta["repetition_id"]
            )[0]
            _, coords26 = archive.read_2d(seq, 1)
            coords18 = to_common_skeleton(coords26)
            normalized = normalize_2d_sequence(coords18)
            pelvis = medoid["feat"]["pelvis_trajectory"][:, 0]
            bottom_rep = int(np.nanargmin(pelvis))
            source_frame = int(meta["frame_range"][0] + bottom_rep)
            row = metrics(normalized[source_frame])
            prep_end = int(medoid["bounds"].get("준비", [0, 0])[1])
            prep_count = prep_end if prep_end > 0 else min(5, int(meta["sequence_length"]))
            standing_rows = [
                metrics(normalized[int(meta["frame_range"][0]) + offset])
                for offset in range(prep_count)
            ]
            standing_knee = float(np.median([item["knee_avg"] for item in standing_rows]))
            standing_hip = float(np.median([item["hip_avg"] for item in standing_rows]))
            knee3d = float(medoid["feat"]["knee_flexion_angle"][bottom_rep].mean() * 180.0)
            hip3d = float(medoid["feat"]["hip_flexion_angle"][bottom_rep].mean() * 180.0)
            standing_knee_3d = float(np.median(
                medoid["feat"]["knee_flexion_angle"][:prep_count].mean(axis=1) * 180.0
            ))
            standing_hip_3d = float(np.median(
                medoid["feat"]["hip_flexion_angle"][:prep_count].mean(axis=1) * 180.0
            ))
            rows.append(row)
            details.append({
                "id": meta["medoid_id"], "bottom_rep_frame": bottom_rep,
                "source_frame": source_frame, **row,
                "knee_3d": knee3d, "knee_delta_3d_minus_2d": knee3d - row["knee_avg"],
                "hip_3d": hip3d, "hip_delta_3d_minus_2d": hip3d - row["hip_avg"],
                "standing_knee_2d": standing_knee,
                "standing_hip_2d": standing_hip,
                "knee_excursion_2d": standing_knee - row["knee_avg"],
                "hip_excursion_2d": standing_hip - row["hip_avg"],
                "knee_excursion_3d": standing_knee_3d - knee3d,
                "hip_excursion_3d": standing_hip_3d - hip3d,
                "knee_excursion_3d_minus_2d": (standing_knee_3d - knee3d) - (standing_knee - row["knee_avg"]),
                "hip_excursion_3d_minus_2d": (standing_hip_3d - hip3d) - (standing_hip - row["hip_avg"]),
            })
    print(json.dumps({
        "normal_reference_details": details,
        "normal_reference_distribution": summary(rows),
        "live_geometry_available": {name: live_metrics(points) for name, points in LIVE.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
