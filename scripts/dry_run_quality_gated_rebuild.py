"""Dry-run quality-gated reference rebuild. Never writes output/reference_db."""
from __future__ import annotations
import csv, hashlib, json, sys, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from ai_trainer.actor_split import load_all_air_squat_sequences
from ai_trainer.aihub_zip import AiHubZip
from ai_trainer.clustering import kmedoids, pairwise_dtw_distance_matrix, sequence_feature_matrix
from ai_trainer.common_skeleton import to_common_skeleton
from ai_trainer.lifting_model import TemporalLiftingNet
from ai_trainer.offline_pose_benchmark import OfflinePoseBenchmark
from ai_trainer.phase_features import extract_phase_features
from ai_trainer.phase_segmentation import segment_phases
from ai_trainer.reference_pipeline import build_ground_truth_reference, build_operational_reference
from ai_trainer.reference_quality_audit import audit_sequence
from scripts.build_reference_db import (TL_ZIP, VL_ZIP, CLASSES, N_CANDIDATES_PER_CLASS,
    K_MEDOIDS, DTW_RADIUS, SEED, sample_candidates)

def file_state(path):
    p=Path(path); return {"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"mtime_ns":p.stat().st_mtime_ns,"size":p.stat().st_size}
def write_csv(path, rows):
    if not rows:return
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ident(origin,obj): return (origin,obj.seq.error_type,obj.seq.level,obj.seq.actor,str(obj.seq.rep))
def select_medoids(items):
    d=pairwise_dtw_distance_matrix([x["dtw"] for x in items],radius=DTW_RADIUS)
    idx,_=kmedoids(d,k=min(K_MEDOIDS,len(items)),seed=SEED)
    return [items[i] for i in idx]

def main():
    total_start=time.perf_counter(); man_path=ROOT/"output/reference_db/manifest.json"; npz_path=ROOT/"output/reference_db/sequences.npz"
    before={str(p):file_state(p) for p in (man_path,npz_path)}
    out=ROOT/"output/diagnostics/quality_gated_reference_rebuild"/datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/"configs/dtw_feature_weights.json").read_text(encoding="utf-8"));bc=cfg["baseline_calibration"]
    hip_min=float(bc["hip_average_min_deg"]);knee_min=float(bc["knee_average_min_deg"])
    manifest=json.loads(man_path.read_text(encoding="utf-8"))["entries"]
    current=defaultdict(list)
    for e in manifest:
        if e["tier"]=="operational":current[e["class_label"]].append((e["origin_zip"],e["class_label"],e["difficulty_level"],e["actor_id"],str(e["repetition_id"])))
    zips={"TL":AiHubZip(TL_ZIP),"VL":AiHubZip(VL_ZIP)};all_seq=load_all_air_squat_sequences(TL_ZIP,VL_ZIP)
    quality_rows=[];reason_rows=[];pools={};t0=time.perf_counter()
    for cls in CLASSES:
        pool=[]
        for obj in sample_candidates(all_seq,cls,N_CANDIDATES_PER_CLASS,seed=SEED):
            ref=build_ground_truth_reference(zips[obj.origin],obj.seq,obj.origin)
            if ref is None:continue
            pf=extract_phase_features(ref.coords);bounds=segment_phases(pf)
            qa=audit_sequence(ref.coords,hip_standing_min=hip_min,knee_standing_min=knee_min)
            _,p2=zips[obj.origin].read_2d(obj.seq,1);start,end=ref.frame_range
            item={"obj":obj,"ref":ref,"bounds":bounds,"qa":qa,"identity":ident(obj.origin,obj),
                  "raw2d":to_common_skeleton(p2[start:end+1]),
                  "dtw":sequence_feature_matrix(pf.pelvis_height,pf.knee_flexion_deg,pf.hip_flexion_deg)}
            pool.append(item)
            cause="NONE"
            if qa["quality"]!="PASS":
                _,full=zips[obj.origin].read_3d(obj.seq);full=to_common_skeleton(full)
                before_crop=full[:start];after_crop=full[end+1:]
                def standing(segment):
                    if len(segment)<3:return False
                    a=audit_sequence(segment if len(segment)>=8 else np.repeat(segment[:1],8,axis=0),hip_standing_min=hip_min,knee_standing_min=knee_min)
                    return a["start_standing_like"] or a["end_standing_like"]
                cause="ANNOTATION_CROP" if standing(before_crop) or standing(after_crop) else "SOURCE_OR_UNKNOWN"
            row={"class":cls,"origin":obj.origin,"actor":obj.seq.actor,"level":obj.seq.level,"rep":obj.seq.rep,
                 "frame_start":start,"frame_end":end,"frames":len(ref.coords),"quality":qa["quality"],
                 "reasons":";".join(qa["reasons"]),"root_cause":cause}
            quality_rows.append(row)
            for reason in qa["reasons"]:reason_rows.append({"reason":reason,"class":cls,"actor":obj.seq.actor,"rep":obj.seq.rep})
        pools[cls]=pool
    reconstruction_seconds=time.perf_counter()-t0
    reproduction=[];clean={};t0=time.perf_counter()
    for cls,pool in pools.items():
        reproduced=select_medoids(pool);passed=[x for x in pool if x["qa"]["quality"]=="PASS"]
        if len(passed)<4: raise RuntimeError(f"{cls}: only {len(passed)} PASS candidates")
        clean[cls]=select_medoids(passed)
        cur=set(current[cls]);rep={x["identity"] for x in reproduced}
        reproduction.append({"class":cls,"expected":40,"reconstructed":len(pool),"current_ids":json.dumps(sorted(map(list,cur)),ensure_ascii=False),
            "reproduced_ids":json.dumps(sorted(map(list,rep)),ensure_ascii=False),"exact_match":len(cur&rep)})
    selection_seconds=time.perf_counter()-t0
    model=TemporalLiftingNet(n_joints=18,hidden=128);model.load_state_dict(torch.load(ROOT/"output/lifting_baseline/model_best.pt",map_location="cpu"));model.eval()
    virtual_entries=[];virtual_arrays={};virtual_2d={};clean_quality=[];comparisons=[];t0=time.perf_counter()
    for cls,selected in clean.items():
        clean_ids=[]
        for rank,item in enumerate(selected):
            op=build_operational_reference(zips[item["obj"].origin],item["obj"].seq,item["obj"].origin,model,torch.device("cpu"))
            key=f"virtual_{rank}_{item['obj'].seq.actor}_rep{item['obj'].seq.rep}";array_key=key+"__operational"
            virtual_arrays[array_key]=op.coords;virtual_2d[key]=item["raw2d"];clean_ids.append(item["identity"])
            qa=audit_sequence(op.coords,hip_standing_min=hip_min,knee_standing_min=knee_min)
            clean_quality.append({"class":cls,"sequence_id":key,"quality":qa["quality"],"reasons":";".join(qa["reasons"])})
            virtual_entries.append({"medoid_id":key,"class_label":cls,"actor_id":item["obj"].seq.actor,
                "difficulty_level":item["obj"].seq.level,"repetition_id":str(item["obj"].seq.rep),"origin_zip":item["obj"].origin,
                "frame_range":list(op.frame_range),"sequence_length":len(op.coords),"phase_boundaries":item["bounds"].as_dict(),
                "tier":"operational","array_key":array_key})
        cur=set(current[cls]);new=set(clean_ids);comparisons.append({"class":cls,"current":json.dumps(sorted(map(list,cur)),ensure_ascii=False),"clean":json.dumps(sorted(map(list,new)),ensure_ascii=False),"changed_count":4-len(cur&new)})
    lifting_seconds=time.perf_counter()-t0
    if any(x["quality"]!="PASS" for x in clean_quality): raise RuntimeError("clean operational medoid failed quality recheck")
    virtual_dir=out/"virtual_reference";virtual_dir.mkdir();(virtual_dir/"manifest.json").write_text(json.dumps({"classes":CLASSES,"entries":virtual_entries},ensure_ascii=False,indent=2),encoding="utf-8")
    np.savez_compressed(virtual_dir/"sequences.npz",**virtual_arrays)
    benchmark=OfflinePoseBenchmark(ROOT,virtual_2d,db_dir=virtual_dir);t0=time.perf_counter();clean_summary,clean_rows=benchmark.run(out/"clean_benchmark");benchmark_seconds=time.perf_counter()-t0
    latest=max((p for p in (ROOT/"output/diagnostics/offline_benchmark").iterdir() if (p/"summary.json").exists()),key=lambda p:p.name)
    original=json.loads((latest/"summary.json").read_text(encoding="utf-8"));metric_rows=[];class_rows=[]
    for method,data in clean_summary["results"].items():
        om=original["results"][method]["metrics"];cm=data["metrics"]
        metric_rows.append({"method":method,"original_accuracy":om["accuracy"],"clean_accuracy":cm["accuracy"],"accuracy_delta":cm["accuracy"]-om["accuracy"],"original_macro_f1":om["macro_f1"],"clean_macro_f1":cm["macro_f1"],"macro_f1_delta":cm["macro_f1"]-om["macro_f1"]})
        for cls in CLASSES:class_rows.append({"method":method,"class":cls,"original_recall":om["per_class"][cls]["recall"],"clean_recall":cm["per_class"][cls]["recall"]})
    for z in zips.values():z.close()
    after={str(p):file_state(p) for p in (man_path,npz_path)};db_unchanged=before==after
    counts={c:dict(Counter(x["qa"]["quality"] for x in pools[c])) for c in CLASSES}
    summary={"candidate_counts":{c:len(pools[c]) for c in CLASSES},"candidate_quality":counts,"reason_counts":dict(Counter(r["reason"] for r in reason_rows)),
        "root_cause_counts":dict(Counter(r["root_cause"] for r in quality_rows if r["root_cause"]!="NONE")),"reproduction":reproduction,
        "clean_quality":dict(Counter(x["quality"] for x in clean_quality)),"metrics":metric_rows,
        "performance_seconds":{"reconstruction_and_audit":reconstruction_seconds,"selection":selection_seconds,"lifting":lifting_seconds,"clean_benchmark":benchmark_seconds,"total":time.perf_counter()-total_start},
        "reference_db_before":before,"reference_db_after":after,"reference_db_unchanged":db_unchanged,"production_runtime_cost":0}
    write_csv(out/"candidate_quality.csv",quality_rows);write_csv(out/"candidate_quality_reasons.csv",reason_rows);write_csv(out/"reproduction_check.csv",reproduction);write_csv(out/"current_vs_clean_medoids.csv",comparisons);write_csv(out/"clean_reference_quality.csv",clean_quality);write_csv(out/"benchmark_original_vs_clean.csv",metric_rows);write_csv(out/"class_metrics_original_vs_clean.csv",class_rows)
    (out/"virtual_reference_manifest.json").write_text(json.dumps({"classes":CLASSES,"entries":virtual_entries},ensure_ascii=False,indent=2),encoding="utf-8");np.savez_compressed(out/"virtual_reference_sequences.npz",**virtual_arrays)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");(out/"report.md").write_text("# Dry-run Quality-Gated Reference Rebuild\n\nReference DB changed: NO\n",encoding="utf-8")
    print(out);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0

if __name__=="__main__":raise SystemExit(main())
