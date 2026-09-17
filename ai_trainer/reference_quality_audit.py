"""Offline-only completeness audit for operational reference sequences."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .angle_domain_diagnostic import frame_angles
from .offline_pose_benchmark import confusion_and_metrics


def _window_stats(values, n):
    x=np.asarray(values,float); n=max(1,min(int(n),len(x)))
    return {"first":float(x[0]),"first3":float(np.median(x[:min(3,len(x))])),
            "first5":float(np.median(x[:min(5,len(x))])),"first10pct":float(np.median(x[:max(1,int(np.ceil(len(x)*.1)))])),
            "last":float(x[-1]),"last3":float(np.median(x[-min(3,len(x)):])),
            "last5":float(np.median(x[-min(5,len(x)):])),
            "last10pct":float(np.median(x[-max(1,int(np.ceil(len(x)*.1))):]))}


def audit_angle_trajectory(hip, knee, *, hip_standing_min, knee_standing_min):
    hip=np.asarray(hip,float);knee=np.asarray(knee,float)
    if hip.ndim!=1 or knee.shape!=hip.shape or len(hip)<8: raise ValueError("angle trajectories must be matching finite 1D arrays with >= 8 frames")
    if not np.isfinite(hip).all() or not np.isfinite(knee).all(): raise ValueError("angle trajectories must be finite")
    n=len(hip); edge=max(3,int(np.ceil(n*.1)))
    start_hip=float(np.median(hip[:min(3,n)]));start_knee=float(np.median(knee[:min(3,n)]))
    end_hip=float(np.median(hip[-min(3,n):]));end_knee=float(np.median(knee[-min(3,n):]))
    combined=(hip+knee)/2; bottom=int(np.argmin(combined)); hip_bottom=int(np.argmin(hip));knee_bottom=int(np.argmin(knee))
    standing_score=np.minimum(hip/hip_standing_min,knee/knee_standing_min)
    best_standing=int(np.argmax(standing_score)); best_hip=float(hip[best_standing]);best_knee=float(knee[best_standing])
    start_ok=start_hip>=hip_standing_min and start_knee>=knee_standing_min
    end_ok=end_hip>=hip_standing_min and end_knee>=knee_standing_min
    excursion_start=np.array([start_hip-hip.min(),start_knee-knee.min()]); excursion_end=np.array([end_hip-hip.min(),end_knee-knee.min()]); excursion_best=np.array([best_hip-hip.min(),best_knee-knee.min()])
    bottom_exists=bool(max(excursion_best)>=15.0); internal=.1*n<=bottom<=.9*n
    descent=bool(bottom>=edge and max(excursion_start)>=15.0);ascent=bool(bottom<n-edge and max(excursion_end)>=15.0)
    # Count well-separated local valleys to flag likely multi-REP crops.
    valleys=[i for i in range(1,n-1) if combined[i]<=combined[i-1] and combined[i]<combined[i+1] and max(best_hip-hip[i],best_knee-knee[i])>=15]
    separated=[]
    for i in valleys:
        if not separated or i-separated[-1]>=max(4,int(.2*n)):separated.append(i)
        elif combined[i]<combined[separated[-1]]:separated[-1]=i
    reasons=[]
    if not start_ok: reasons.append("START_NOT_STANDING")
    if not end_ok: reasons.append("END_NOT_STANDING")
    if not descent: reasons.append("DESCENT_MISSING")
    if not bottom_exists: reasons.append("BOTTOM_MISSING_OR_FLAT")
    if not ascent: reasons.append("ASCENT_MISSING")
    if not internal: reasons.append("BOTTOM_AT_BOUNDARY")
    if len(separated)>1: reasons.append("MULTIPLE_REP_CANDIDATE")
    if abs(hip_bottom-knee_bottom)>.2*n: reasons.append("HIP_KNEE_BOTTOM_MISMATCH")
    full=start_ok and end_ok and descent and bottom_exists and ascent and internal and len(separated)<=1
    severe=not bottom_exists or (not start_ok and not descent) or (not end_ok and not ascent)
    quality="PASS" if full and not reasons else "SEVERE" if severe else "SUSPICIOUS"
    return {"start_standing_like":start_ok,"end_standing_like":end_ok,"has_descent":descent,"has_bottom":bottom_exists,
            "has_ascent":ascent,"returns_to_standing":end_ok and ascent,"bottom_not_at_boundary":internal,
            "full_rep_candidate":full,"quality":quality,"reasons":reasons,"bottom_frame":bottom,
            "bottom_position":float(bottom/max(n-1,1)),"hip_bottom_frame":hip_bottom,"knee_bottom_frame":knee_bottom,
            "bottom_timing_difference":abs(hip_bottom-knee_bottom),"rep_valley_count":len(separated),
            "standing_estimates":{"first_frame":[float(hip[0]),float(knee[0])],"first_robust":[start_hip,start_knee],
                "start_end_more_standing":[max(start_hip,end_hip),max(start_knee,end_knee)],"global_candidate":[best_hip,best_knee]},
            "excursions":{"first_frame":[float(hip[0]-hip.min()),float(knee[0]-knee.min())],
                "first_n_robust":excursion_start.tolist(),"start_end_more_standing":np.maximum(excursion_start,excursion_end).tolist(),
                "global_standing_candidate":excursion_best.tolist()},"hip":_window_stats(hip,edge),"knee":_window_stats(knee,edge),
            "bottom_angles":[float(hip.min()),float(knee.min())]}


def audit_sequence(coords, *, hip_standing_min, knee_standing_min):
    p=np.asarray(coords,float)
    if p.ndim!=3 or p.shape[1:]!=(18,3):raise ValueError("sequence shape must be (T,18,3)")
    angles=[frame_angles(x) for x in p]
    return audit_angle_trajectory([x["hip_avg"] for x in angles],[x["knee_avg"] for x in angles],
                                  hip_standing_min=hip_standing_min,knee_standing_min=knee_standing_min)


def sensitivity(rows, method, quality="PASS"):
    selected=[r for r in rows if r["quality"]==quality]
    labels=list(dict.fromkeys(r["actual_class"] for r in rows))
    if not selected:return {"evaluated":0,"excluded":len(rows),"class_support":{},"accuracy":None,"macro_f1":None}
    _,m=confusion_and_metrics([r["actual_class"] for r in selected],[r["methods"][method]["predicted_class"] for r in selected],labels)
    return {"evaluated":len(selected),"excluded":len(rows)-len(selected),"class_support":dict(Counter(r["actual_class"] for r in selected)),"accuracy":m["accuracy"],"macro_f1":m["macro_f1"]}


__all__=["audit_angle_trajectory","audit_sequence","sensitivity"]
