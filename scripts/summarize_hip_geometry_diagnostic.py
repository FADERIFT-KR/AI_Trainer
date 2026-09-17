"""Print a compact summary of a diagnostic-only hip geometry JSONL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
reps = [row for row in rows if row.get("record_type") == "completed_rep"]
print("REPS", len(reps))
print("REF", json.dumps(rows[0]["geometry_distributions"], ensure_ascii=False))
for row in reps:
    g = row["hip_geometry"]
    a, b = g["original"], g["hip_adjusted"]
    print(json.dumps({
        "rep": row["production"]["rep"], "prep": g["live_prep"], "bottom": g["live_bottom"],
        "A_k2": a["knee"]["bottom_2d"], "A_k3": a["knee"]["bottom_3d"],
        "A_h2": a["hip"]["bottom_2d"], "A_h3": a["hip"]["bottom_3d"],
        "A_kdelta": a["consistency"]["knee_delta"], "A_hdelta": a["consistency"]["hip_delta"],
        "A_pass": a["consistency"]["pass"], "B_k3": b["knee"]["bottom_3d"],
        "B_h3": b["hip"]["bottom_3d"], "B_kdelta": b["consistency"]["knee_delta"],
        "B_hdelta": b["consistency"]["hip_delta"], "B_pass": b["consistency"]["pass"],
        "improvement": g["improvement"], "trigger": g["production_trigger"],
    }, ensure_ascii=False))
for ratio in ("hip_width/shoulder_width", "hip_width/torso_length", "hip_width/leg_length"):
    values = np.array([row["hip_geometry"]["live_bottom"][ratio] for row in reps])
    knee = np.array([row["hip_geometry"]["original"]["knee"]["bottom_distortion"] for row in reps])
    hip = np.array([row["hip_geometry"]["original"]["hip"]["bottom_distortion"] for row in reps])
    print("CORR", ratio, "knee", np.corrcoef(values, knee)[0, 1], "hip", np.corrcoef(values, hip)[0, 1])
