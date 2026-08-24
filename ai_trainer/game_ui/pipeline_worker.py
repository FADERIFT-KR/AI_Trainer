"""Qt 워커 스레드: 카메라 + MediaPipe(ms.choe) + Common Skeleton + Lifting +
OnlineSquatSession(이 브랜치)을 한 스레드에서 순차 실행하고 신호로 내보낸다.

ms.choe의 `live_pose.worker.CameraPoseWorker`와 같은 패턴(카메라 오픈 로직,
FPS 스무딩, 에러 처리)을 따르되 파이프라인 뒷단(Common Skeleton 매핑 ->
2D->3D Lifting -> 정규화 -> Online Phase/DTW)을 추가한다. `FrameProcessor`를
그대로 쓰지 않고 detector.process()를 직접 호출하는 이유는 표시용 프레임과
이 브랜치의 DTW 파이프라인이 **같은 PoseObservation**(2중 추론 방지)을
공유해야 하기 때문이다.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector, PoseBackendError
from ai_trainer.live_pose.render import draw_2d_pose
from ai_trainer.live_pose.worker import CameraConfig, _open_camera
from ai_trainer.lifting_model import TemporalLiftingNet
from ai_trainer.online_dtw import OnlineSquatSession
from ai_trainer.reference_db_io import load_reference_db

from .framing_check import check_framing, guide_box as compute_guide_box

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
LIFTING_CKPT = ROOT / "output" / "lifting_baseline" / "model_best.pt"
WEIGHTS_CFG_PATH = ROOT / "configs" / "dtw_feature_weights.json"
DB_DIR = ROOT / "output" / "reference_db"
OFFLINE_REPORT_PATH = ROOT / "output" / "dtw_eval" / "offline_eval_report.json"


@dataclass(frozen=True)
class PipelineStatus:
    video_bgr: np.ndarray
    fps: float
    pose_found: bool
    mean_confidence: float
    n_frozen: int
    framing_ok: bool
    framing_message: str
    phase: str | None
    rep_count: int
    partial_distance: dict | None  # {"phase":..., "distance_by_class": {...}}
    completed_rep: object | None  # ai_trainer.online_dtw.RepResult


class SquatPipelineWorker(QThread):
    status_ready = pyqtSignal(object)  # PipelineStatus
    status_changed = pyqtSignal(str)
    fatal_error = pyqtSignal(str)

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, config: CameraConfig | None = None, parent=None):
        super().__init__(parent)
        self.model_path = Path(model_path).resolve()
        self.config = config or CameraConfig()

    def run(self) -> None:
        capture = None
        detector = None
        try:
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("OpenCV가 설치되지 않았습니다. pip install -r requirements.txt") from error

            self.status_changed.emit("모델을 불러오는 중…")
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            lifting_model = TemporalLiftingNet(n_joints=18, hidden=128)
            lifting_model.load_state_dict(torch.load(LIFTING_CKPT, map_location=device))
            lifting_model.to(device).eval()

            weights_cfg = json.loads(WEIGHTS_CFG_PATH.read_text(encoding="utf-8"))
            db = load_reference_db(DB_DIR)
            score_calib = None
            if OFFLINE_REPORT_PATH.exists():
                score_calib = json.loads(OFFLINE_REPORT_PATH.read_text(encoding="utf-8"))["score_calibration"]

            session = OnlineSquatSession(
                model=lifting_model, device=device, db_operational=db["operational"],
                weights_cfg=weights_cfg, score_calib=score_calib,
            )

            from .pose_bridge import CommonSkeletonBridge

            bridge = CommonSkeletonBridge(min_visibility=self.config.confidence)

            detector = MediaPipePoseDetector(
                self.model_path,
                min_detection_confidence=self.config.confidence,
                min_presence_confidence=self.config.confidence,
                min_tracking_confidence=self.config.confidence,
            )

            self.status_changed.emit("카메라를 여는 중…")
            capture = _open_camera(cv2, self.config)
            self.status_changed.emit("실행 중")

            previous_time = time.perf_counter()
            smoothed_fps = 0.0
            consecutive_failures = 0

            while not self.isInterruptionRequested():
                success, frame_bgr = capture.read()
                if not success or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        raise RuntimeError("카메라 프레임을 연속으로 읽지 못했습니다.")
                    self.msleep(10)
                    continue
                consecutive_failures = 0

                display_bgr = np.ascontiguousarray(frame_bgr[:, ::-1] if self.config.mirror else frame_bgr)
                observation = detector.process(np.ascontiguousarray(display_bgr[:, :, ::-1]))

                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                smoothed_fps = instantaneous_fps if smoothed_fps == 0.0 else smoothed_fps * 0.90 + instantaneous_fps * 0.10

                phase = None
                partial = None
                completed = None
                mean_conf = 0.0
                n_frozen = 0
                pose_found = observation is not None
                framing_ok = False
                framing_message = "카메라 앞에 서주세요"
                h, w = display_bgr.shape[:2]
                gbox = compute_guide_box(w, h)

                if pose_found:
                    video_bgr = draw_2d_pose(display_bgr, observation.image_landmarks)
                    framing = check_framing(observation.image_landmarks, w, h)
                    framing_ok = framing.ok
                    framing_message = framing.message

                    box_color = (90, 220, 90) if framing.ok else (60, 60, 240)
                    cv2.rectangle(video_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), box_color, 2)
                    if framing.body_box is not None:
                        cv2.rectangle(video_bgr, (framing.body_box[0], framing.body_box[1]), (framing.body_box[2], framing.body_box[3]), (0, 200, 255), 1)

                    if framing_ok:
                        # 화각/거리/정면 여부가 학습 데이터(AI Hub camera1)와 맞을 때만 phase/DTW 파이프라인 진행.
                        # 그렇지 않으면 잘못된 프레임이 session의 phase 상태기계에 섞여 들어가지 않도록 건너뛴다.
                        common2d, frozen_mask, mean_conf = bridge.update(observation.image_landmarks, w, h)
                        n_frozen = int(frozen_mask.sum())
                        status = session.push_frame(common2d)
                        if status is not None and status.get("status") == "ok":
                            phase = status["phase"]
                            partial = status["partial_distance"]
                            if status["event"] == "rep_end":
                                completed = status["completed_rep"]
                else:
                    video_bgr = display_bgr.copy()
                    cv2.rectangle(video_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (60, 60, 240), 2)

                self.status_ready.emit(
                    PipelineStatus(
                        video_bgr=video_bgr,
                        fps=smoothed_fps,
                        pose_found=pose_found,
                        mean_confidence=mean_conf,
                        n_frozen=n_frozen,
                        framing_ok=framing_ok,
                        framing_message=framing_message,
                        phase=phase,
                        rep_count=len(session.completed_reps),
                        partial_distance=partial,
                        completed_rep=completed,
                    )
                )
        except (PoseBackendError, RuntimeError, ValueError, OSError) as error:
            self.fatal_error.emit(str(error))
        except Exception as error:  # noqa: BLE001
            self.fatal_error.emit(f"파이프라인 처리 중 예기치 않은 오류: {error}")
        finally:
            if capture is not None:
                capture.release()
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass


__all__ = ["SquatPipelineWorker", "PipelineStatus"]
