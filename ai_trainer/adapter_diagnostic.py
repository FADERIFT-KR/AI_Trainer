"""In-memory A/B lifting diagnostic; never feeds production classification."""
from __future__ import annotations
import numpy as np
import torch

from .common_skeleton import COMMON_JOINT_NAMES
from .geometry_adapter import adapt_mediapipe_to_training_domain
from .lifting_dataset import WINDOW_T
from .features import extract_all_features

I={n:i for i,n in enumerate(COMMON_JOINT_NAMES)}; HALF=WINDOW_T//2

def angle(a,b,c):
    u,v=a-b,c-b
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-8),-1,1))))

def angles(frame):
    kh=[angle(frame[I['LHip']],frame[I['LKnee']],frame[I['LAnkle']]),angle(frame[I['RHip']],frame[I['RKnee']],frame[I['RAnkle']])]
    hh=[angle(frame[I['Neck']],frame[I['LHip']],frame[I['LKnee']]),angle(frame[I['Neck']],frame[I['RHip']],frame[I['RKnee']])]
    return {'knee_lr':kh,'hip_lr':hh,'knee':sum(kh)/2,'hip':sum(hh)/2}

def geometry(frame):
    d=lambda a,b: float(np.linalg.norm(frame[I[a]]-frame[I[b]]))
    torso=max(d('Neck','Hip'),1e-8); hip=max(d('LHip','RHip'),1e-8)
    return {'hip_width/torso':d('LHip','RHip')/torso,'shoulder_width/torso':d('LShoulder','RShoulder')/torso,
      'left_thigh/torso':d('LHip','LKnee')/torso,'right_thigh/torso':d('RHip','RKnee')/torso,
      'left_shin/torso':d('LKnee','LAnkle')/torso,'right_shin/torso':d('RKnee','RAnkle')/torso,
      'knee_width/hip_width':d('LKnee','RKnee')/hip,'ankle_width/hip_width':d('LAnkle','RAnkle')/hip}

def geometry_ratios(frame):
    """Scale/translation-invariant widths used only by live diagnostics."""
    d=lambda a,b: float(np.linalg.norm(frame[I[a]]-frame[I[b]]))
    hip=max(d('LHip','RHip'),1e-8); shoulder=max(d('LShoulder','RShoulder'),1e-8)
    torso=max(d('Neck','Hip'),1e-8)
    leg=max(0.5*((d('LHip','LKnee')+d('LKnee','LAnkle'))+
                 (d('RHip','RKnee')+d('RKnee','RAnkle'))),1e-8)
    return {'hip_width/shoulder_width':hip/shoulder,
            'hip_width/torso_length':hip/torso,
            'hip_width/leg_length':hip/leg,
            'knee_width/hip_width':d('LKnee','RKnee')/hip,
            'ankle_width/hip_width':d('LAnkle','RAnkle')/hip}

def adjust_hip_width_copy(sequence, target_hip_over_torso):
    """Move each hip and its whole leg chain together on a diagnostic copy."""
    src=np.asarray(sequence); out=src.astype(np.float64,copy=True)
    for frame in out:
        center=(frame[I['LHip']]+frame[I['RHip']])/2.0
        axis=frame[I['RHip']]-frame[I['LHip']]
        axis=axis/max(float(np.linalg.norm(axis)),1e-8)
        torso=float(np.linalg.norm(frame[I['Neck']]-frame[I['Hip']]))
        half=float(target_hip_over_torso)*torso/2.0
        for side,sign in (('L',-1.0),('R',1.0)):
            new_hip=center+sign*axis*half
            delta=new_hip-frame[I[f'{side}Hip']]
            for joint in (f'{side}Hip',f'{side}Knee',f'{side}Ankle',f'{side}Heel',f'{side}BigToe'):
                frame[I[joint]]+=delta
    return out.astype(src.dtype,copy=False)

def lift_sequence(raw, model, device, scale2d, rotation, scale3d):
    out=[]
    for t in range(len(raw)):
        lo=max(0,t-HALF); win=raw[lo:t+HALF+1]
        if len(win)<WINDOW_T:
            if lo==0: win=np.concatenate([np.repeat(win[:1],WINDOW_T-len(win),axis=0),win])
            else: win=np.concatenate([win,np.repeat(win[-1:],WINDOW_T-len(win),axis=0)])
        norm=(win-win[:,I['Hip']:I['Hip']+1])/scale2d
        with torch.no_grad(): p=model(torch.from_numpy(norm[None].astype(np.float32)).to(device))[0].cpu().numpy()
        out.append(np.einsum('ij,pj->pi',rotation.T,p/scale3d))
    return np.asarray(out)

def summarize(raw, pose3d, detector_bottom, visibility):
    a2=[angles(f) for f in raw]; a3=[angles(f) for f in pose3d]
    standing_n=min(5,len(raw)); s2={j:float(np.median([x[j] for x in a2[:standing_n]])) for j in ('knee','hip')}; s3={j:float(np.median([x[j] for x in a3[:standing_n]])) for j in ('knee','hip')}
    progress=np.array([(s2['knee']-x['knee'])+(s2['hip']-x['hip']) for x in a2]); bottom=int(np.argmax(progress))
    result={'length':len(raw),'actual_bottom':bottom,'detector_bottom':detector_bottom,'bottom_offset':detector_bottom-bottom,'visibility':visibility,'geometry':geometry(raw[bottom]),'bottom_lr_2d':a2[bottom]}
    for j in ('knee','hip'):
        b2=a2[bottom][j]; b3=a3[bottom][j]; e2=s2[j]-min(x[j] for x in a2); e3=s3[j]-min(x[j] for x in a3)
        result[j]={'standing_2d':s2[j],'bottom_2d':b2,'excursion_2d':e2,'standing_3d':s3[j],'bottom_3d':b3,'excursion_3d':e3,'bottom_distortion':b3-b2,'excursion_error':abs(e3-e2)}
    feat=extract_all_features(pose3d); result['pelvis_min']=float(np.min(feat['pelvis_trajectory'][:,0]))
    return result

def run_adapter_ab(session,start,end,detector_bottom,visibility):
    raw=np.asarray(session.raw2d_buffer[start:end+1]); adapted=adapt_mediapipe_to_training_domain(raw)
    raw3=np.asarray(session.aligned_seq[session._arr_idx(start):session._arr_idx(end)+1])
    adapted3=lift_sequence(adapted,session.model,session.device,session.scale2d,session.R_body,session.scale3d)
    a=summarize(raw,raw3,detector_bottom,visibility); b=summarize(adapted,adapted3,detector_bottom,visibility)
    for j in ('knee','hip'): b[j]['distortion_improvement']=abs(a[j]['bottom_distortion'])-abs(b[j]['bottom_distortion'])
    return {'raw':a,'adapted':b}

def run_hip_width_ab(session,start,end,detector_bottom,visibility,target_hip_over_torso,semantic_cfg):
    """Controlled hip-width-only A/B. The B sequence never enters production."""
    raw=np.asarray(session.raw2d_buffer[start:end+1])
    adjusted=adjust_hip_width_copy(raw,target_hip_over_torso)
    raw3=np.asarray(session.aligned_seq[session._arr_idx(start):session._arr_idx(end)+1])
    adjusted3=lift_sequence(adjusted,session.model,session.device,session.scale2d,session.R_body,session.scale3d)
    a=summarize(raw,raw3,detector_bottom,visibility)
    b=summarize(adjusted,adjusted3,detector_bottom,visibility)
    kmin=float(semantic_cfg['knee_3d_minus_2d_excursion_min_deg']); kmax=float(semantic_cfg['knee_3d_minus_2d_excursion_max_deg'])
    hmin=float(semantic_cfg['hip_3d_minus_2d_excursion_min_deg']); hmax=float(semantic_cfg['hip_3d_minus_2d_excursion_max_deg'])
    for row in (a,b):
        kd=row['knee']['excursion_3d']-row['knee']['excursion_2d']
        hd=row['hip']['excursion_3d']-row['hip']['excursion_2d']
        row['consistency']={'knee_delta':kd,'hip_delta':hd,
                            'knee_pass':kmin<=kd<=kmax,'hip_pass':hmin<=hd<=hmax,
                            'pass':kmin<=kd<=kmax and hmin<=hd<=hmax}
    bottom=a['actual_bottom']; prep=slice(0,min(5,len(raw)))
    raw_prep={k:float(np.median([geometry_ratios(f)[k] for f in raw[prep]])) for k in geometry_ratios(raw[0])}
    norm=(raw-raw[:,I['Hip']:I['Hip']+1])/session.scale2d
    norm_prep={k:float(np.median([geometry_ratios(f)[k] for f in norm[prep]])) for k in geometry_ratios(norm[0])}
    return {'target_hip_width/torso_length':float(target_hip_over_torso),
            'live_prep':raw_prep,'live_bottom':geometry_ratios(raw[bottom]),
            'lifting_input_prep':norm_prep,'normalization_ratio_delta':{k:norm_prep[k]-raw_prep[k] for k in raw_prep},
            'original':a,'hip_adjusted':b,
            'improvement':{'knee_bottom_distortion':abs(a['knee']['bottom_distortion'])-abs(b['knee']['bottom_distortion']),
                           'hip_bottom_distortion':abs(a['hip']['bottom_distortion'])-abs(b['hip']['bottom_distortion']),
                           'consistency_changed':a['consistency']['pass']!=b['consistency']['pass']}}

__all__=['run_adapter_ab','run_hip_width_ab','geometry_ratios','adjust_hip_width_copy']
