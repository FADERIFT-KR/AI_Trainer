"""Independent three-view capture. Space confirms five repetitions; Escape aborts.

No game/REP/classifier changes. Saves raw mirrored camera video, matched 2D/lifting
panels, and observation timestamps. Video frames are held at 30fps, not fabricated
camera observations. Lifting uses timestamp interpolation and a fixed initial
eight-observation torso scale (streaming convention, unlike full-clip CSV scale).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES, COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR
from ai_trainer.game_ui.pose_bridge import _DIRECT_MP_IDX
from ai_trainer.lifting_model import TemporalLiftingNet
from ai_trainer.normalization import body_axes
from ai_trainer.render import draw_skeleton_panel, fit_transform

STAGES = ('front', 'side_a', 'side_b')


def common_xy(landmarks, width, height):
    a = np.asarray(landmarks, dtype=float)
    if a.shape != (33, 4) or not np.isfinite(a).all():
        return None, False
    xy = a[:, :2] * [width, height]
    points = np.stack([(xy[23]+xy[24])/2 if n == 'Hip' else
                       (xy[11]+xy[12])/2 if n == 'Neck' else
                       xy[_DIRECT_MP_IDX[n]] for n in COMMON_JOINT_NAMES])
    indices = list(_DIRECT_MP_IDX.values())
    valid = bool((a[indices, 3] >= .4).all() and
                 ((a[indices, :2] >= 0) & (a[indices, :2] <= 1)).all())
    return points, valid


def latest_window(times, poses, valid, origin):
    """Nine 30Hz samples, only adjacent valid observations, max gap .5 seconds."""
    last = origin + np.floor((times[-1]-origin)*30 + 1e-7)/30
    grid = last - np.arange(8, -1, -1)/30
    if grid[0] < times[0]-1e-8:
        return None
    window = []
    for t in grid:
        b = int(np.searchsorted(times, t))
        if b == len(times) and abs(t-times[-1]) < 1e-8:
            b -= 1
        if b < len(times) and abs(times[b]-t) < 1e-8:
            if not valid[b]:
                return None
            window.append(poses[b])
        else:
            a = b-1
            if a < 0 or b >= len(times) or not (valid[a] and valid[b]) or times[b]-times[a] > .5:
                return None
            alpha = (t-times[a])/(times[b]-times[a])
            window.append((1-alpha)*poses[a]+alpha*poses[b])
    return np.stack(window), float(grid[4])


class HeldVideo:
    """Causal zero-order hold on a 30Hz output clock; original timestamps in JSONL."""
    def __init__(self, path, size):
        if path.exists():
            raise FileExistsError(path)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
        if not self.writer.isOpened():
            raise RuntimeError(f'Cannot open video writer: {path}')
        self.previous = None
        self.next_index = 0

    def push(self, frame, elapsed):
        while self.previous is not None and self.next_index/30 < elapsed:
            self.writer.write(self.previous)
            self.next_index += 1
        self.previous = frame.copy()

    def close(self):
        if self.previous is not None:
            self.writer.write(self.previous)
            self.next_index += 1
        self.writer.release()


class Stage:
    def __init__(self, directory, name, size, model):
        self.name, self.model = name, model
        self.raw = HeldVideo(directory / f'{name}_raw.mp4', size)
        self.skeleton = HeldVideo(directory / f'{name}_lifting.mp4', (1440, 540))
        self.stream = (directory / f'{name}_trace.jsonl').open('x', encoding='utf-8')
        self.times, self.poses, self.valid = [], [], []
        self.calibration, self.axes = [], []
        self.scale = self.rotation = None
        self.count = self.predictions = 0
        self.start = None
        self.tf2 = fit_transform(np.array([[[0, 0], list(size)]]), 480, 420, margin=30, flip_y=False)
        self.tf3 = fit_transform(np.array([[[-650, -1050], [650, 850]]]), 480, 420, margin=30, flip_y=True)

    def process(self, image, observation, timestamp, observation_id):
        if self.start is None:
            self.start = timestamp
        points, good = (None, False) if observation is None else common_xy(observation.image_landmarks, image.shape[1], image.shape[0])
        self.times.append(timestamp)
        self.poses.append(points)
        self.valid.append(good)
        if good and self.scale is None:
            self.calibration.append(float(np.linalg.norm(points[17]-points[16])))
            if len(self.calibration) == 8:
                self.scale = max(float(np.median(self.calibration)), 1e-6)
        result = latest_window(self.times, self.poses, self.valid, self.start) if self.scale is not None else None
        prediction = center = input2d = display = None
        if result is not None:
            window, center = result
            input2d = window[4]
            normalized = (window-window[:, 16:17])/self.scale
            with torch.inference_mode():
                prediction = self.model(torch.from_numpy(normalized[None].astype('float32')))[0].numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError('Non-finite Lifting prediction')
            centered = prediction-prediction[16:17]
            if self.rotation is None:
                self.axes.append(body_axes(centered))
                if len(self.axes) >= 5:
                    mean = np.mean(self.axes, axis=0)
                    lateral = mean[:, 0]/np.linalg.norm(mean[:, 0])
                    forward = np.cross(lateral, mean[:, 1])
                    norm = np.linalg.norm(forward)
                    if norm < 1e-6:
                        raise RuntimeError('Degenerate initial body axes')
                    forward /= norm
                    self.rotation = np.stack([lateral, np.cross(forward, lateral), forward], axis=1)
            if self.rotation is not None:
                display = centered @ self.rotation
            self.predictions += 1
        canvas = np.zeros((540, 1440, 3), np.uint8)
        cv2.putText(canvas, f'{self.name.upper()} | Perform 5 squats, then SPACE | ESC: stop/save', (16, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 1, cv2.LINE_AA)
        warning = 'Lifting estimate only. Side views are OUT OF TRAINING VIEW. No automatic REP counting.'
        cv2.putText(canvas, warning, (16, 55), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,200,255), 1, cv2.LINE_AA)
        if input2d is not None and display is not None:
            panels = [(self.tf2(input2d), '2D input at sample center', 'Interpolated XY; +Y down'),
                      (self.tf3(display[:,[0,1]]), 'Lifting 3D: body XY', 'Fixed initial rotation; +Y up'),
                      (self.tf3(display[:,[2,1]]), 'Lifting 3D: body ZY', 'Same center time; millimeters')]
            for j,(p,title,footer) in enumerate(panels):
                draw_skeleton_panel(canvas, (480*j,70),480,420,p,title,footer,COMMON_BONE_INDEX_PAIRS,COMMON_BONE_COLORS_BGR)
        else:
            cv2.putText(canvas, 'Stand still, full body visible: calibrating or invalid 2D window', (35,250), cv2.FONT_HERSHEY_SIMPLEX,.8,(200,200,255),1,cv2.LINE_AA)
        center_text = 'none' if center is None else f'{center-self.start:.3f}s'
        cv2.putText(canvas, f'Observation {observation_id} t={timestamp-self.start:.3f}s | 3D center={center_text} | SPACE confirms your 5 reps', (16,518),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1,cv2.LINE_AA)
        record = dict(observation_id=observation_id, timestamp=timestamp, elapsed=timestamp-self.start,
                      mirrored=True, orientation='mediapipe_input', image_size=[image.shape[1],image.shape[0]],
                      image_landmarks=None if observation is None else observation.image_landmarks.tolist(),
                      world_landmarks=None if observation is None else observation.world_landmarks.tolist(),
                      lifting_input_valid=good, lifting_center_timestamp=center,
                      lifting_input_center_xy=None if input2d is None else input2d.tolist(),
                      lifting_xyz_mm=None if prediction is None else prediction.tolist(),
                      scale2d=self.scale, display_rotation=None if self.rotation is None else self.rotation.tolist())
        self.stream.write(json.dumps(record)+'\n'); self.stream.flush()
        self.raw.push(image, timestamp-self.start)
        self.skeleton.push(canvas, timestamp-self.start)
        self.count += 1
        return canvas

    def close(self, confirmed):
        self.raw.close();self.skeleton.close();self.stream.close()
        return dict(stage=self.name, user_confirmed_five_reps=confirmed, automatically_counted=False,
                    observations=self.count, predictions=self.predictions, video_frames=self.raw.next_index,
                    observation_duration=0 if len(self.times)<2 else self.times[-1]-self.times[0])


def main():
    from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector
    from ai_trainer.live_pose.worker import CameraConfig, _open_camera
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    directory=args.output or ROOT/'output'/'diagnostics'/('three_views_'+uuid4().hex)
    directory.mkdir(parents=True,exist_ok=False)
    print('OUTPUT='+str(directory),flush=True)
    torch.set_num_threads(4)
    model=TemporalLiftingNet(n_joints=18,hidden=128,dilations=(1,2,1))
    model.load_state_dict(torch.load(ROOT/'output/lifting_baseline/model_best.pt',map_location='cpu',weights_only=True),strict=True)
    model.eval()
    summary=dict(stages=[], mode='manual_space_confirms_5', video_fps=30,
                 video_timing='previous observation held; not 30fps camera acquisition',
                 lifting='30Hz interpolated 9-frame window, max gap .5s, visibility >=.4, no extrapolation',
                 normalization='first 8 valid observations torso median per stage, not whole-clip CSV normalization')
    detector=capture=stage=None
    try:
        detector=MediaPipePoseDetector(ROOT/'models/pose_landmarker_full.task',min_detection_confidence=.4,min_presence_confidence=.4,min_tracking_confidence=.4)
        capture=_open_camera(cv2,CameraConfig())
        cv2.namedWindow('Three views - lifting',cv2.WINDOW_NORMAL);cv2.resizeWindow('Three views - lifting',1440,540)
        cv2.namedWindow('Camera - original mirrored',cv2.WINDOW_NORMAL);cv2.resizeWindow('Camera - original mirrored',640,360)
        index=0;observation_id=0;ready=time.perf_counter()+10;last_space=-10;failures=0
        while index<3:
            ok,frame=capture.read();timestamp=time.perf_counter()
            if not ok:
                failures+=1
                if failures>=30:raise RuntimeError('Camera read failed 30 times')
                continue
            failures=0;image=np.ascontiguousarray(frame[:,::-1]);cv2.imshow('Camera - original mirrored',image)
            if timestamp<ready:
                preview=np.zeros((540,1440,3),np.uint8)
                cv2.putText(preview,f'GET READY: {STAGES[index].upper()} | {max(1,int(ready-timestamp)+1)} seconds',(50,220),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
                cv2.putText(preview,'Full body visible. Stand still first. Then 5 squats and SPACE.',(50,290),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,220,255),1)
                cv2.imshow('Three views - lifting',preview)
            else:
                if stage is None:
                    stage=Stage(directory,STAGES[index],(image.shape[1],image.shape[0]),model)
                    print('STAGE='+STAGES[index],flush=True)
                observation=detector.process(np.ascontiguousarray(image[:,:,::-1]))
                canvas=stage.process(image,observation,timestamp,observation_id)
                cv2.imshow('Three views - lifting',canvas)
                observation_id+=1
            key=cv2.waitKey(1)&255
            if key==27:break
            if key==32 and stage is not None and timestamp-last_space>2:
                summary['stages'].append(stage.close(True));stage=None;index+=1
                last_space=timestamp;ready=time.perf_counter()+10
                print('CONFIRMED_STAGES='+str(index),flush=True)
            if cv2.getWindowProperty('Three views - lifting',cv2.WND_PROP_VISIBLE)<1:break
    finally:
        if stage is not None:summary['stages'].append(stage.close(False))
        if capture is not None:capture.release()
        if detector is not None:detector.close()
        cv2.destroyAllWindows()
        (directory/'session.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('SAVED='+str(directory),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    main()
