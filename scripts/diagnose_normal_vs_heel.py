"""Offline normal-vs-heel feature separation diagnostic."""
from __future__ import annotations
import csv,json,sys,time
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from ai_trainer.actor_split import load_all_air_squat_sequences
from ai_trainer.aihub_zip import AiHubZip
from ai_trainer.common_skeleton import to_common_skeleton
from ai_trainer.dtw_compare import phase_aware_weighted_dtw,resolve_weights
from ai_trainer.feature_separation import separation,semantic_counts,remove_one
from ai_trainer.features import FEATURE_NAMES,extract_all_features
from ai_trainer.heel_semantic import heel_motion_evidence
from ai_trainer.offline_pose_benchmark import OfflinePoseBenchmark,derivative_features,geometry_vector,robust_location_scale,confusion_and_metrics
from ai_trainer.phase_features import extract_phase_features
from ai_trainer.phase_segmentation import segment_phases
from ai_trainer.reference_pipeline import build_ground_truth_reference
from ai_trainer.reference_quality_audit import audit_sequence
from scripts.build_reference_db import TL_ZIP,VL_ZIP,N_CANDIDATES_PER_CLASS,SEED,sample_candidates

NORMAL="정상";HEEL="발뒤꿈치오류";LABELS=[NORMAL,HEEL]
def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def predict_temporal(samples,cfg,weights,derivative=False):
    rows=[]
    for q in samples:
        dist={}
        for label in LABELS:
            vals=[]
            for r in samples:
                if r['label']!=label or r['id']==q['id']:continue
                qf=derivative_features(q['features']) if derivative else q['features'];rf=derivative_features(r['features']) if derivative else r['features']
                vals.append(phase_aware_weighted_dtw(qf,q['bounds'],rf,r['bounds'],weights,cfg)['total'])
            dist[label]=min(vals)
        pred=min(dist,key=dist.get);rows.append({'id':q['id'],'actual':q['label'],'predicted':pred,'normal_distance':dist[NORMAL],'heel_distance':dist[HEEL],'margin':abs(dist[NORMAL]-dist[HEEL])})
    return rows
def metrics(rows):
    matrix,m=confusion_and_metrics([r['actual'] for r in rows],[r['predicted'] for r in rows],LABELS)
    return matrix,m
def nearest_scalar(rows,key):
    out=[]
    for qi,q in enumerate(rows):
        train=[r for i,r in enumerate(rows) if i!=qi];loc,scale=robust_location_scale(np.stack([r[key] for r in train]));d={}
        for label in LABELS:d[label]=min(float(np.linalg.norm((q[key]-r[key])/scale)) for r in train if r['label']==label)
        out.append({'actual':q['label'],'predicted':min(d,key=d.get)})
    return metrics(out)

def main():
    started=time.perf_counter();out=ROOT/'output/diagnostics/normal_vs_heel_feature_separation'/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/dtw_feature_weights.json').read_text(encoding='utf-8'));bc=cfg['baseline_calibration'];hs=cfg['heel_semantic_validation']
    zips={'TL':AiHubZip(TL_ZIP),'VL':AiHubZip(VL_ZIP)};allseq=load_all_air_squat_sequences(TL_ZIP,VL_ZIP);source=[]
    for label in LABELS:
        for obj in sample_candidates(allseq,label,N_CANDIDATES_PER_CLASS,seed=SEED):
            ref=build_ground_truth_reference(zips[obj.origin],obj.seq,obj.origin)
            if ref is None:continue
            qa=audit_sequence(ref.coords,hip_standing_min=bc['hip_average_min_deg'],knee_standing_min=bc['knee_average_min_deg'])
            if qa['quality']!='PASS':continue
            _,p2=zips[obj.origin].read_2d(obj.seq,1);s,e=ref.frame_range;raw2d=to_common_skeleton(p2[s:e+1]);ev=heel_motion_evidence(raw2d,no_max=hs['no_evidence_max'],yes_min=hs['positive_evidence_min'])
            geom,names=geometry_vector(ref.coords,raw2d);f=extract_all_features(ref.coords);scalar=dict(zip(names,geom))
            for name,val in f.items():
                x=np.asarray(val,float);delta=np.diff(x,axis=0);scalar[name+'_range']=float(np.mean(np.ptp(x,axis=0)));scalar[name+'_mean_abs_derivative']=float(np.mean(np.abs(delta)));scalar[name+'_peak_derivative']=float(np.max(np.linalg.norm(delta,axis=1)))
            source.append({'id':f'{label}_{obj.origin}_{obj.seq.actor}_{obj.seq.level}_{obj.seq.rep}','label':label,'actor':obj.seq.actor,'geometry':geom[:12],'heelvec':geom[12:],'scalar':scalar,'heel_evidence':ev['value'],'heel_verdict':ev['verdict']})
    scalar_names=sorted(set.intersection(*(set(x['scalar']) for x in source)));stats=[]
    for name in scalar_names:
        a=[x['scalar'][name] for x in source if x['label']==NORMAL];b=[x['scalar'][name] for x in source if x['label']==HEEL];s=separation(a,b)
        stats.append({'feature':name,'domain':'2D' if 'heel_' in name and name.endswith('_2d') else '3D','normal_n':len(a),'heel_n':len(b),'normal_median':s['normal']['median'],'heel_median':s['heel']['median'],'cohens_d':s['cohens_d'],'robust_effect':s['robust_effect'],'raw_auc':s['raw_auc'],'directionless_separation':s['directionless_separation'],'overlap':s['overlap']})
    stats.sort(key=lambda r:(r['directionless_separation'],-r['overlap']),reverse=True)
    semantic=[]
    for label in LABELS:
        vals=[x['heel_evidence'] for x in source if x['label']==label];counts=semantic_counts(vals,hs['no_evidence_max'],hs['positive_evidence_min'])
        semantic.append({'class':label,'n':len(vals),**counts,**{k+'_pct':v/len(vals) for k,v in counts.items()}})
    latest=max((p for p in (ROOT/'output/diagnostics/quality_gated_reference_rebuild').iterdir() if (p/'virtual_reference').exists()),key=lambda p:p.name)
    bench=OfflinePoseBenchmark(ROOT,db_dir=latest/'virtual_reference');clean=[s for s in bench.samples if s['label'] in LABELS]
    weights=resolve_weights(bench.config,bench.config['default_profile'],None);full=predict_temporal(clean,bench.config,weights);ddtw=predict_temporal(clean,bench.config,weights,True)
    ablations=[]
    for removed in ['NONE',*FEATURE_NAMES]:
        w=weights if removed=='NONE' else remove_one(weights,removed);pred=predict_temporal(clean,bench.config,w);matrix,m=metrics(pred)
        ablations.append({'removed_feature':removed,'accuracy':m['accuracy'],'normal_recall':m['per_class'][NORMAL]['recall'],'heel_recall':m['per_class'][HEEL]['recall'],'normal_to_heel':int(matrix[0,1]),'macro_f1':m['macro_f1']})
    false=[];contrib=[]
    for method,preds,deriv in [('DTW',full,False),('DDTW',ddtw,True)]:
        for pr in preds:
            if pr['actual']!=NORMAL or pr['predicted']!=HEEL:continue
            q=next(x for x in clean if x['id']==pr['id'])
            source_match=next((x for x in source if x['label']==q['label'] and x['actor']==q['entry']['actor_id'] and x['id'].endswith('_'+str(q['entry']['repetition_id']))),None)
            ev='UNKNOWN' if source_match is None else source_match['heel_verdict']
            false.append({'sample':pr['id'],'method':method,'normal_distance':pr['normal_distance'],'heel_distance':pr['heel_distance'],'margin':pr['margin'],'heel_evidence':ev,'heel_value':None if source_match is None else source_match['heel_evidence']})
            best={}
            for label in LABELS:
                scored=[]
                for r in clean:
                    if r['label']!=label or r['id']==q['id']:continue
                    qf=derivative_features(q['features']) if deriv else q['features'];rf=derivative_features(r['features']) if deriv else r['features']
                    res=phase_aware_weighted_dtw(qf,q['bounds'],rf,r['bounds'],weights,bench.config);scored.append((res['total'],res['per_feature_contrib']))
                best[label]=min(scored,key=lambda x:x[0])[1]
            for feature in FEATURE_NAMES:contrib.append({'sample':pr['id'],'method':method,'feature':feature,'normal_contribution':best[NORMAL][feature],'heel_contribution':best[HEEL][feature],'heel_advantage':best[NORMAL][feature]-best[HEEL][feature]})
    heel_matrix,heel_m=nearest_scalar(source,'heelvec');geo_matrix,geo_m=nearest_scalar(source,'geometry');full_matrix,full_m=metrics(full);ddtw_matrix,ddtw_m=metrics(ddtw)
    actor=[];allgeom=np.stack([x['geometry'] for x in source]);_,scale=robust_location_scale(allgeom)
    groups={k:[] for k in ('same_actor_same_class','different_actor_same_class','same_actor_different_class','different_actor_different_class')}
    for i,a in enumerate(source):
        for b in source[i+1:]:
            key=('same_actor_' if a['actor']==b['actor'] else 'different_actor_')+('same_class' if a['label']==b['label'] else 'different_class');groups[key].append(float(np.linalg.norm((a['geometry']-b['geometry'])/scale)))
    for k,v in groups.items():actor.append({'group':k,'n':len(v),'median':None if not v else float(np.median(v)),'mean':None if not v else float(np.mean(v))})
    ddtw_groups=[]
    for pr in ddtw:
        if pr['actual']==NORMAL:ddtw_groups.append(pr)
    summary={'dataset':dict(Counter(x['label'] for x in source)),'thresholds':hs,'semantic':semantic,'two_class':{'heel_only_2d':{'matrix':heel_matrix.tolist(),'metrics':heel_m},'geometry_3d':{'matrix':geo_matrix.tolist(),'metrics':geo_m},'DTW':{'matrix':full_matrix.tolist(),'metrics':full_m},'DDTW':{'matrix':ddtw_matrix.tolist(),'metrics':ddtw_m}},'top_features':stats[:10],'poor_features':stats[-10:],'performance_seconds':time.perf_counter()-started,'production_runtime_cost':0}
    write_csv(out/'feature_statistics.csv',stats);write_csv(out/'feature_separability.csv',stats);write_csv(out/'heel_semantic_distribution.csv',semantic);write_csv(out/'false_heel_samples.csv',false);write_csv(out/'dtw_feature_contributions.csv',contrib);write_csv(out/'dtw_single_feature_ablation.csv',ablations);write_csv(out/'ddtw_feature_group_analysis.csv',ddtw_groups);write_csv(out/'heel_only_2class_results.csv',[{'metric':k,'value':v} for k,v in heel_m.items() if k!='per_class']);write_csv(out/'geometry_2class_results.csv',[{'metric':k,'value':v} for k,v in geo_m.items() if k!='per_class']);write_csv(out/'normal_vs_heel_confusion.csv',[{'method':k,'matrix':json.dumps(v['matrix'])} for k,v in summary['two_class'].items()]);write_csv(out/'actor_domain_analysis.csv',actor)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');(out/'report.md').write_text('# Normal vs Heel-Error Feature Separation Diagnostic\n\nProduction unchanged.\n',encoding='utf-8')
    for z in zips.values():z.close()
    print(out);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
