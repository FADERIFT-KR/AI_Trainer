"""실제 운영용 2D Pose Estimator (MediaPipe Pose, Task API) 래퍼.

주의: 설치된 mediapipe(1.0.0)는 구버전 `mp.solutions.pose`가 아니라 신버전
Task API(`mp.tasks.vision.PoseLandmarker`)만 제공한다. 33개 랜드마크 구성 자체는
동일하다. 모델 파일은 `models/pose_landmarker_lite.task`(Google 공식 배포)를 사용한다.

MediaPipe는 좌표를 이미지 폭/높이로 각각 나눈 [0,1] 정규화 좌표를 반환하므로,
가로세로 비율이 다른 이미지(예: 640x480)에서 그대로 쓰면 사람 형태가 찌그러진다.
반드시 (width, height)를 다시 곱해 **실제 픽셀 좌표**로 되돌린 뒤 사용해야
AI Hub pixel-space 2D CSV와 동일한 기하학적 의미를 가진다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

# MediaPipe PoseLandmark 인덱스 (Task API도 legacy와 동일한 33점 토폴로지)
MP_LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]


@dataclass
class PoseFrameResult:
    landmarks_px: np.ndarray  # (33,2) 픽셀 좌표
    visibility: np.ndarray  # (33,) 0~1
    detected: bool


class MediaPipePoseEstimator:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"PoseLandmarker 모델이 없습니다: {model_path}\n"
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
                "float16/latest/pose_landmarker_lite.task 에서 받아 저장하세요."
            )
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseFrameResult:
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1  # MediaPipe VIDEO 모드는 timestamp가 단조증가해야 함
        self._last_ts_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return PoseFrameResult(landmarks_px=np.zeros((33, 2)), visibility=np.zeros(33), detected=False)

        lms = result.pose_landmarks[0]
        px = np.array([[lm.x * w, lm.y * h] for lm in lms])  # 정규화 좌표 -> 픽셀 좌표 (종횡비 보존 필수)
        vis = np.array([lm.visibility for lm in lms])
        return PoseFrameResult(landmarks_px=px, visibility=vis, detected=True)

    def close(self) -> None:
        self._landmarker.close()
