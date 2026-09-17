"""Offline-only feature separation statistics."""
from __future__ import annotations
import numpy as np

def validate_values(values):
    x=np.asarray(values,float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError("feature values must be finite 1D with N>=2")
    return x
def distribution_stats(values):
    x=validate_values(values);q=np.percentile(x,[0,10,25,50,75,90,100])
    return dict(zip(("min","p10","p25","median","p75","p90","max","mean","std","iqr"),map(float,(*q,np.mean(x),np.std(x),q[4]-q[2]))))|{"n":len(x)}
def rank_auc(normal,heel):
    a=validate_values(normal);b=validate_values(heel);return float(np.mean(b[:,None]>a[None,:])+.5*np.mean(b[:,None]==a[None,:]))
def overlap_coefficient(normal,heel,bins=20):
    a=validate_values(normal);b=validate_values(heel);lo=min(a.min(),b.min());hi=max(a.max(),b.max())
    if hi==lo:return 1.0
    ha,e=np.histogram(a,bins=bins,range=(lo,hi),density=True);hb,_=np.histogram(b,bins=bins,range=(lo,hi),density=True)
    return float(np.minimum(ha,hb).sum()*(e[1]-e[0]))
def separation(normal,heel):
    a=validate_values(normal);b=validate_values(heel);sa=distribution_stats(a);sb=distribution_stats(b)
    pooled=np.sqrt((np.var(a)+np.var(b))/2);piqr=(sa["iqr"]+sb["iqr"])/2;delta=sb["median"]-sa["median"]
    auc=rank_auc(a,b)
    return {"normal":sa,"heel":sb,"median_difference":delta,"normalized_median_difference":delta/max(abs(sa["median"]),1e-8),
            "cohens_d":float((np.mean(b)-np.mean(a))/max(pooled,1e-8)),"robust_effect":float(delta/max(piqr,1e-8)),
            "raw_auc":auc,"directionless_separation":max(auc,1-auc),"overlap":overlap_coefficient(a,b)}
def semantic_counts(values,no_max,yes_min):
    out={"NO":0,"AMBIGUOUS":0,"YES":0}
    for v in validate_values(values):out["NO" if v<=no_max else "YES" if v>=yes_min else "AMBIGUOUS"]+=1
    return out
def remove_one(weights,name):
    out=dict(weights);out[name]=0.0;return out

__all__=["distribution_stats","rank_auc","overlap_coefficient","separation","semantic_counts","remove_one"]
