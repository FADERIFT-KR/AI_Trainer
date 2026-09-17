"""Run the offline Reference Quality Audit and phase completeness diagnostic."""
from __future__ import annotations

import csv,json,sys,time
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from ai_trainer.angle_domain_diagnostic import frame_angles
from ai_trainer.reference_quality_audit import audit_sequence,sensitivity
from scripts.benchmark_pose_methods import load_operational_2d


def write_csv(path,rows):
    if not rows:return
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    manifest=json.loads((ROOT/"output/reference_db/manifest.json").read_text(encoding="utf-8"))["entries"]
    arrays=np.load(ROOT/"output/reference_db/sequences.npz");cfg=json.loads((ROOT/"configs/dtw_feature_weights.json").read_text(encoding="utf-8"))
    base=cfg["baseline_calibration"]; hip_min=float(base["hip_average_min_deg"]);knee_min=float(base["knee_average_min_deg"])
    raw2d=load_operational_2d(ROOT); stamp=datetime.now().strftime("%Y%m%d_%H%M%S");out=ROOT/"output/diagnostics/reference_quality_audit"/stamp
    (out/"trajectories").mkdir(parents=True,exist_ok=False); started=time.perf_counter(); rows=[]
    for e in manifest:
        if e["tier"]!="operational":continue
        coords=np.asarray(arrays[e["array_key"]]); result=audit_sequence(coords,hip_standing_min=hip_min,knee_standing_min=knee_min)
        a=[frame_angles(x) for x in coords]
        traj=[]
        for i,x in enumerate(a):traj.append({"frame":i,"normalized_time":i/max(len(a)-1,1),"left_hip":x["hip_lr"][0],"right_hip":x["hip_lr"][1],"avg_hip":x["hip_avg"],"left_knee":x["knee_lr"][0],"right_knee":x["knee_lr"][1],"avg_knee":x["knee_avg"]})
        write_csv(out/"trajectories"/(e["medoid_id"]+".csv"),traj)
        rows.append({"class":e["class_label"],"sequence_id":e["medoid_id"],"actor":e["actor_id"],"rep":e["repetition_id"],
            "origin":e["origin_zip"],"frame_range":json.dumps(e["frame_range"]),"frames":len(coords),"shape_2d":str(raw2d[e["medoid_id"]].shape),"shape_3d":str(coords.shape),"finite":bool(np.isfinite(coords).all()),
            **{k:result[k] for k in ("start_standing_like","end_standing_like","has_descent","has_bottom","has_ascent","returns_to_standing","bottom_not_at_boundary","full_rep_candidate","quality")},
            "reasons":";".join(result["reasons"]),"bottom_position":result["bottom_position"],"bottom_timing_difference":result["bottom_timing_difference"],
            "start_hip":result["hip"]["first10pct"],"start_knee":result["knee"]["first10pct"],"end_hip":result["hip"]["last10pct"],"end_knee":result["knee"]["last10pct"],
            "bottom_hip":result["bottom_angles"][0],"bottom_knee":result["bottom_angles"][1],"excursion_definitions":json.dumps(result["excursions"])})
    audit_seconds=time.perf_counter()-started
    bench_root=ROOT/"output/diagnostics/offline_benchmark";latest=max((p for p in bench_root.iterdir() if (p/"sample_results.json").exists()),key=lambda p:p.name)
    benchmark=json.loads((latest/"sample_results.json").read_text(encoding="utf-8"));quality={r["sequence_id"]:r for r in rows};joined=[]
    for b in benchmark:
        q=quality[b["sample_id"]]; joined.append({"sample_id":b["sample_id"],"actual_class":b["actual_class"],"quality":q["quality"],"reason":q["reasons"],"DTW_prediction":b["methods"]["DTW"]["predicted_class"],"DTW_correct":b["methods"]["DTW"]["correct"],"DDTW_prediction":b["methods"]["DDTW"]["predicted_class"],"DDTW_correct":b["methods"]["DDTW"]["correct"],"Geometry_prediction":b["methods"]["Geometry"]["predicted_class"],"Geometry_correct":b["methods"]["Geometry"]["correct"]})
        b["quality"]=q["quality"]
    sens={m:sensitivity(benchmark,m) for m in ("DTW","DDTW","Geometry")}
    quality_counts=dict(Counter(r["quality"] for r in rows)); class_counts={c:dict(Counter(r["quality"] for r in rows if r["class"]==c)) for c in dict.fromkeys(r["class"] for r in rows)}
    summary={"audit_name":"Reference Quality Audit + Phase Completeness Diagnostic","standing_rule":{"source":"configs/dtw_feature_weights.json baseline_calibration reference-derived averages","hip_min":hip_min,"knee_min":knee_min},"operational_count":len(rows),"quality_counts":quality_counts,"class_quality":class_counts,"audit_seconds":audit_seconds,"ms_per_sequence":audit_seconds*1000/len(rows),"benchmark_source":str(latest),"pass_test_only_sensitivity":sens,"candidate_pass_only":{"available":False,"reason":"prior benchmark artifact stores class minima, not per-reference candidate distances; no production/reference mutation performed"},"source_full_audit":{"available":False,"reason":"manifest contains only medoids and cluster counts, not identities of all 160 sampled candidates"}}
    write_csv(out/"operational_reference_quality.csv",rows);write_csv(out/"phase_completeness.csv",rows);write_csv(out/"benchmark_quality_join.csv",joined);write_csv(out/"suspicious_references.csv",[r for r in rows if r["quality"]!="PASS"])
    write_csv(out/"source_quality_summary.csv",[{"status":"operational 16 only","reason":summary["source_full_audit"]["reason"]}])
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Reference Quality Audit","",f"Standing rule: hip >= {hip_min:.2f}, knee >= {knee_min:.2f} (existing reference-derived baseline)","",f"PASS={quality_counts.get('PASS',0)}, SUSPICIOUS={quality_counts.get('SUSPICIOUS',0)}, SEVERE={quality_counts.get('SEVERE',0)}","","Production/reference DB unchanged."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(out);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
