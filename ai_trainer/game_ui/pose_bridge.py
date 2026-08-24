"""ms.choe의 `live_pose.PoseObservation`(MediaPipe 33관절) -> 이 브랜치의
Common Skeleton(18노드) 변환 + 저신뢰 관절 freeze 처리.

`ai_trainer.pose_estimator`/`mediapipe_mapping`(feature/dtw-pipeline에서 제거됨)과
같은 역할이지만, ms.choe의 `PoseObservation.image_landmarks`
((33,4) float32 array: 정규화 x,y, z, visibility) 형식에 맞춰 새로 작성했다.
"""
from __future__ import annotations

import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES

# MediaPipe PoseLandmark 인덱스 (Task API, 33점 — live_pose.render.POSE_CONNECTIONS와 동일 토폴로지)
_MP_INDEX = {
    "NOSE": 0, "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7, "RIGHT_EAR": 8, "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12, "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24, "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28, "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}

_DIRECT_MAP = {
    "LShoulder": "LEFT_SHOULDER", "RShoulder": "RIGHT_SHOULDER",
    "LElbow": "LEFT_ELBOW", "RElbow": "RIGHT_ELBOW",
    "LWrist": "LEFT_WRIST", "RWrist": "RIGHT_WRIST",
    "LHip": "LEFT_HIP", "RHip": "RIGHT_HIP",
    "LKnee": "LEFT_KNEE", "RKnee": "RIGHT_KNEE",
    "LAnkle": "LEFT_ANKLE", "RAnkle": "RIGHT_ANKLE",
    "LHeel": "LEFT_HEEL", "RHeel": "RIGHT_HEEL",
    "LBigToe": "LEFT_FOOT_INDEX", "RBigToe": "RIGHT_FOOT_INDEX",
}
_DIRECT_MP_IDX = {common: _MP_INDEX[mp_name] for common, mp_name in _DIRECT_MAP.items()}


class CommonSkeletonBridge:
    """세션 동안 직전 신뢰값을 유지하며 image_landmarks -> Common Skeleton 변환."""

    def __init__(self, min_visibility: float = 0.5):
        self.min_visibility = min_visibility
        self.last_good = np.zeros((len(COMMON_JOINT_NAMES), 2))
        self.has_good = np.zeros(len(COMMON_JOINT_NAMES), dtype=bool)

    def update(self, image_landmarks: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, float]:
        """image_landmarks: (33,4) [x,y,z,visibility], x/y는 [0,1] 정규화.

        반환: (common 18x2 픽셀좌표(결측은 freeze), frozen_mask(18,), mean_confidence)
        """
        px_all = image_landmarks[:, :2] * np.array([width, height])
        vis_all = image_landmarks[:, 3]

        raw = np.zeros((len(COMMON_JOINT_NAMES), 2))
        conf = np.zeros(len(COMMON_JOINT_NAMES))
        for i, name in enumerate(COMMON_JOINT_NAMES):
            if name == "Hip":
                l, r = _MP_INDEX["LEFT_HIP"], _MP_INDEX["RIGHT_HIP"]
                raw[i] = (px_all[l] + px_all[r]) / 2.0
                conf[i] = min(vis_all[l], vis_all[r])
            elif name == "Neck":
                l, r = _MP_INDEX["LEFT_SHOULDER"], _MP_INDEX["RIGHT_SHOULDER"]
                raw[i] = (px_all[l] + px_all[r]) / 2.0
                conf[i] = min(vis_all[l], vis_all[r])
            else:
                idx = _DIRECT_MP_IDX[name]
                raw[i] = px_all[idx]
                conf[i] = vis_all[idx]

        frozen = np.zeros(len(COMMON_JOINT_NAMES), dtype=bool)
        out = self.last_good.copy()
        for i in range(len(COMMON_JOINT_NAMES)):
            if conf[i] >= self.min_visibility:
                out[i] = raw[i]
                self.last_good[i] = raw[i]
                self.has_good[i] = True
            else:
                frozen[i] = True
                if not self.has_good[i]:
                    out[i] = raw[i]

        return out, frozen, float(np.mean(conf))
