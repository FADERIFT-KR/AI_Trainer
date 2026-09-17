#!/usr/bin/env python3
"""Audit-only Normal vs Heel feature analysis for the 404-sequence LOAO run."""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trainer.actor_split import load_all_air_squat_sequences
from ai_trainer.aihub_zip import AiHubZip
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES, to_common_skeleton
from ai_trainer.heel_semantic import heel_motion_evidence
from ai_trainer.reference_pipeline import build_ground_truth_reference
from ai_trainer.two_d_diagnostic import extract_2d_features
from scripts.build_reference_db import TL_ZIP, VL_ZIP

NORMAL = "정상"
HEEL = "발뒤꿈치오류"
UNKNOWN = "자세추정불확실"
I = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
BASELINE = ROOT / "output/diagnostics/actor_disjoint_validation/20260908_155029"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sid(item) -> str:
    s = item.seq
    return f"{item.origin}:{s.error_type}:{s.level}:{s.actor}:rep{s.rep}"


def auc(target, score):
    y, s = np.asarray(target, bool), np.asarray(score, float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos)*len(neg)))


def quantiles(values):
    x = np.asarray(values, float)
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std()), **{f"p{q}": float(np.percentile(x, q)) for q in (10,25,75,90)}}


def wrapped_delta(angle, base):
    return np.abs((angle - base + 90.0) % 180.0 - 90.0)


def features(points: np.ndarray, no_max: float, yes_min: float) -> tuple[dict, dict]:
    p = np.asarray(points, float); n = len(p); prep = min(5, n)
    _, diag = extract_2d_features(p)
    bottom = int(diag["bottom_frame_global"])
    torso = max(float(np.median(np.linalg.norm(p[:prep, I["Neck"]] - p[:prep, I["Hip"]], axis=1))), 1e-6)
    out = {}
    angle_series, heel_disp, toe_gap = {}, {}, {}
    for side in ("L", "R"):
        heel = p[:, I[side+"Heel"]]; toe = p[:, I[side+"BigToe"]]
        vec = toe - heel
        # Image coordinates: +x right, +y down. Acute inclination to horizontal is side invariant.
        angle = np.degrees(np.arctan2(np.abs(vec[:, 1]), np.abs(vec[:, 0]) + 1e-9))
        base = float(np.median(angle[:prep])); dev = wrapped_delta(angle, base)
        hdisp = (heel[:, 1] - float(np.median(heel[:prep, 1]))) / torso
        gap = (heel[:, 1] - toe[:, 1]) / torso
        gbase = float(np.median(gap[:prep]))
        angle_series[side], heel_disp[side], toe_gap[side] = angle, hdisp, gap
        prefix = "left" if side == "L" else "right"
        out.update({
            f"{prefix}_foot_angle_standing": base,
            f"{prefix}_foot_angle_bottom": float(angle[bottom]),
            f"{prefix}_foot_angle_bottom_change": float(wrapped_delta(angle[bottom], base)),
            f"{prefix}_foot_angle_max_deviation": float(dev.max()),
            f"{prefix}_foot_angle_range": float(np.ptp(angle)),
            f"{prefix}_heel_vertical_max_abs": float(np.max(np.abs(hdisp))),
            f"{prefix}_heel_vertical_bottom_abs": float(abs(hdisp[bottom])),
            f"{prefix}_heel_toe_bottom_change_abs": float(abs(gap[bottom]-gbase)),
            f"{prefix}_heel_toe_range": float(np.ptp(gap)),
        })
    for stem in ("foot_angle_bottom_change", "foot_angle_max_deviation", "foot_angle_range",
                 "heel_vertical_max_abs", "heel_vertical_bottom_abs",
                 "heel_toe_bottom_change_abs", "heel_toe_range"):
        a, b = out["left_"+stem], out["right_"+stem]
        out[stem+"_max"] = max(a,b); out[stem+"_mean"] = (a+b)/2; out[stem+"_asymmetry"] = abs(a-b)
    out.update({
        "heel_semantic_raw_score": float(heel_motion_evidence(p, no_max=no_max, yes_min=yes_min)["value"]),
        "heel_ankle_motion_max": max(float(diag["left_heel_ankle_motion"]), float(diag["right_heel_ankle_motion"])),
        "heel_relative_motion": float(diag["heel_relative_motion"]),
        "heel_toe_delta_bottom_max_abs": max(abs(float(diag["left_heel_toe_delta"])), abs(float(diag["right_heel_toe_delta"]))),
        "heel_root_delta_bottom_max_abs": max(abs(float(diag["left_root_centered_heel_delta"])), abs(float(diag["right_root_centered_heel_delta"]))),
    })
    # Phase-aware maximum foot-angle deviation from standing; boundaries are diagnostic only.
    phase_ranges = {"prep": (0, prep), "descent": (prep, max(prep+1, bottom)),
                    "bottom": (max(0,bottom-2), min(n,bottom+3)), "ascent": (bottom, n)}
    for phase, (lo, hi) in phase_ranges.items():
        hi = max(lo+1, min(n,hi)); lo = min(lo,n-1)
        vals=[]
        for side in ("L","R"):
            base=float(np.median(angle_series[side][:prep])); vals.append(float(wrapped_delta(angle_series[side][lo:hi],base).max()))
        out[f"foot_angle_deviation_{phase}_max"] = max(vals)
    evidence = heel_motion_evidence(p, no_max=no_max, yes_min=yes_min)
    return out, evidence


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "output/diagnostics/normal_vs_heel_patch3" / stamp
    outdir.mkdir(parents=True, exist_ok=False)
    cfg=json.loads((ROOT/"configs/dtw_feature_weights.json").read_text(encoding="utf-8"))
    hc=cfg["heel_semantic_validation"]; no_max=float(hc["no_evidence_max"]); yes_min=float(hc["positive_evidence_min"])
    predictions=read_csv(BASELINE/"predictions.csv"); pred={r["source_id"]:r for r in predictions}
    groups=Counter(); enriched=[]; excluded=[]
    zips={"TL":AiHubZip(TL_ZIP),"VL":AiHubZip(VL_ZIP)}
    try:
        items=load_all_air_squat_sequences(TL_ZIP,VL_ZIP)
        for idx,item in enumerate(items,1):
            sample_id=sid(item)
            if sample_id not in pred: continue
            try:
                ref=build_ground_truth_reference(zips[item.origin],item.seq,item.origin)
                _, pts=zips[item.origin].read_2d(item.seq,1); start,end=ref.frame_range
                raw2d=to_common_skeleton(pts[start:end+1]); feat,evidence=features(raw2d,no_max,yes_min)
                row={**pred[sample_id],**feat,"recomputed_heel_verdict":evidence["verdict"]}
                enriched.append(row)
                if row["true_class"] in (NORMAL,HEEL):
                    key=f"GT {row['true_class']} -> Pred {row['final_class']}"; groups[key]+=1
            except Exception as error: excluded.append({"source_id":sample_id,"reason":f"{type(error).__name__}: {error}"})
            if idx%50==0: print(f"[FEATURE] {idx}/{len(items)}",flush=True)
    finally:
        for z in zips.values(): z.close()
    nh=[r for r in enriched if r["true_class"] in (NORMAL,HEEL)]
    metadata=set(predictions[0])|{"recomputed_heel_verdict"}
    feature_names=[k for k in nh[0] if k not in metadata and isinstance(nh[0][k],(int,float,np.floating))]
    metrics=[]; actor_metrics=[]; rng=random.Random(20260916); actors=sorted({r["actor"] for r in nh})
    for name in feature_names:
        normal=[float(r[name]) for r in nh if r["true_class"]==NORMAL]; heel=[float(r[name]) for r in nh if r["true_class"]==HEEL]
        raw=auc([False]*len(normal)+[True]*len(heel),normal+heel)
        direction=1 if raw>=.5 else -1; aware=max(raw,1-raw)
        actor_aucs=[]
        for actor in actors:
            ar=[r for r in nh if r["actor"]==actor]
            if {r["true_class"] for r in ar}=={NORMAL,HEEL}:
                aa=auc([r["true_class"]==HEEL for r in ar],[float(r[name]) for r in ar]); actor_aucs.append(aa if direction>0 else 1-aa)
                actor_metrics.append({"feature":name,"actor":actor,"normal_n":sum(r["true_class"]==NORMAL for r in ar),"heel_n":sum(r["true_class"]==HEEL for r in ar),"auc_direction_aligned":actor_aucs[-1],"normal_median":float(np.median([float(r[name]) for r in ar if r["true_class"]==NORMAL])),"heel_median":float(np.median([float(r[name]) for r in ar if r["true_class"]==HEEL]))})
        boots=[]; by_actor={a:[r for r in nh if r["actor"]==a] for a in actors}
        for _ in range(1000):
            draw=[rng.choice(actors) for _ in actors]; rows=[r for a in draw for r in by_actor[a]]
            a=auc([r["true_class"]==HEEL for r in rows],[float(r[name])*direction for r in rows]); boots.append(a)
        ns,hs=quantiles(normal),quantiles(heel)
        metrics.append({"feature":name,**{f"normal_{k}":v for k,v in ns.items()},**{f"heel_{k}":v for k,v in hs.items()},"raw_auc_heel_high":raw,"heel_direction":"higher" if direction>0 else "lower","direction_aware_auc":aware,"actor_auc_n":len(actor_aucs),"actor_auc_median":float(np.median(actor_aucs)) if actor_aucs else None,"actor_bootstrap_ci_low":float(np.percentile(boots,2.5)),"actor_bootstrap_ci_high":float(np.percentile(boots,97.5)),"interpretability":"high" if any(x in name for x in ("heel_semantic","heel_vertical","foot_angle")) else "medium","camera_sensitivity":"high" if "vertical" in name or "angle" in name else "medium","production_cost":"low"})
    metrics.sort(key=lambda r:r["direction_aware_auc"],reverse=True)
    n2h=[r for r in enriched if r["true_class"]==NORMAL and r["final_class"]==HEEL]
    nunk=[r for r in enriched if r["true_class"]==NORMAL and r["final_class"]==UNKNOWN]
    detail_fields=["actor","source_id","true_class","raw_class","final_class","distance_정상","distance_발뒤꿈치오류","distance_엉덩이하방오류","distance_고관절오류","heel_value","heel_verdict","semantic_reason",*feature_names]
    error_rows=[{k:r.get(k) for k in detail_fields} for r in n2h]
    unknown_rows=[{k:r.get(k) for k in detail_fields} for r in nunk]
    raw_counts=Counter(r["raw_class"] for r in n2h); evidence_counts=Counter(r["heel_verdict"] for r in n2h)
    unknown_reasons=Counter(r["semantic_reason"] for r in nunk)
    # Exploratory leakage-safe equal-weight combination of the two leading non-duplicate direct features.
    best_vertical=next((m["feature"] for m in metrics if "heel_vertical" in m["feature"] and not m["feature"].startswith(("left_","right_"))),None)
    best_angle=next((m["feature"] for m in metrics if "foot_angle" in m["feature"] and not m["feature"].startswith(("left_","right_"))),None)
    combo_features=[f for f in (best_vertical,best_angle) if f is not None]
    combo_scores=[]
    if len(combo_features)==2:
        for held in actors:
            train=[r for r in nh if r["actor"]!=held]; test=[r for r in nh if r["actor"]==held]
            params=[]
            for f in combo_features:
                vals=np.asarray([float(r[f]) for r in train]); med=float(np.median(vals)); scale=max(float(np.percentile(vals,75)-np.percentile(vals,25)),1e-8)
                na=np.median([float(r[f]) for r in train if r["true_class"]==NORMAL]); he=np.median([float(r[f]) for r in train if r["true_class"]==HEEL]); params.append((med,scale,1 if he>=na else -1))
            for r in test: combo_scores.append((r["true_class"]==HEEL,sum(sign*(float(r[f])-med)/scale for f,(med,scale,sign) in zip(combo_features,params))/2))
    combo_auc=auc([x[0] for x in combo_scores],[x[1] for x in combo_scores]) if combo_scores else None
    summary={"name":"Accuracy Patch 3 Pre-Audit: Normal vs Heel Error Actor-Disjoint Feature Analysis","audit_only":True,"baseline":str(BASELINE),"source_count":len(enriched),"excluded":len(excluded),"normal_heel_count":len(nh),"prediction_groups":dict(groups),"thresholds_unchanged":{"heel_no_max":no_max,"heel_yes_min":yes_min},"normal_to_heel":{"count":len(n2h),"raw_class_counts":dict(raw_counts),"heel_evidence_counts":dict(evidence_counts),"primary_mechanism":"RAW DTW selects Heel; semantic guard only blocks NO evidence and never creates Heel"},"normal_unknown":{"count":len(nunk),"reason_counts":dict(unknown_reasons)},"top_features":metrics[:15],"exploratory_actor_disjoint_combination":{"features":combo_features,"auc":combo_auc,"method":"within each held-out actor fold: train-actor median/IQR scaling, direction from train medians, equal-weight mean; no threshold fitted"},"production_changes":"NONE"}
    write_csv(outdir/"feature_metrics.csv",metrics); write_csv(outdir/"normal_to_heel_errors.csv",error_rows); write_csv(outdir/"normal_unknown_errors.csv",unknown_rows); write_csv(outdir/"actor_feature_metrics.csv",actor_metrics); write_csv(outdir/"feature_distributions.csv",[{"feature":m["feature"],"class":"Normal",**{k.replace("normal_",""):v for k,v in m.items() if k.startswith("normal_")}} for m in metrics]+[{"feature":m["feature"],"class":"Heel Error",**{k.replace("heel_",""):v for k,v in m.items() if k.startswith("heel_") and k not in ("heel_direction",)}} for m in metrics]); write_csv(outdir/"excluded_samples.csv",excluded)
    (outdir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (outdir/"README.txt").write_text("Audit only. Production, thresholds, model, and reference DB are unchanged. Foot angle is the acute angle of toe-minus-heel to the image horizontal axis (+x right, +y down); REP features use changes from the first-five-frame standing baseline.\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print(f"[DONE] {outdir}")


if __name__ == "__main__": main()
