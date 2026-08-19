"""MediaPipe 33관절 -> Common Skeleton(18노드) 매핑 + confidence 기반 결측 처리.

claude.md 4장 Common Skeleton Mapping 표의 MediaPipe 쪽 규칙:
  - 직접 대응 관절(어깨/팔꿈치/손목/엉덩이/무릎/발목/뒤꿈치/BigToe≈foot_index)은 그대로 사용
  - Pelvis(Hip) = (LHip+RHip)/2, Neck = (LShoulder+RShoulder)/2 (AI Hub는 직접 제공값을 쓰지만
    MediaPipe에는 없는 관절이라 좌우 평균으로 파생)

결측/저신뢰 관절 처리: 프레임마다 visibility가 임계값 미만이면 이번 프레임 값을 쓰지 않고
**직전에 신뢰도 있었던 값을 유지(freeze)**한다 — 가려짐으로 인한 튀는 값이 lifting 모델에
그대로 들어가는 것을 방지한다. 세션 시작 후 한 번도 신뢰값이 없었던 관절은 0으로 둔다.
"""
from __future__ import annotations

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES
from .pose_estimator import MP_LANDMARK_NAMES

_MP_IDX = {name: i for i, name in enumerate(MP_LANDMARK_NAMES)}

# Common Skeleton 관절 -> MediaPipe 인덱스 (직접 대응되는 것만)
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
_DIRECT_MP_IDX = {common: _MP_IDX[mp_name] for common, mp_name in _DIRECT_MAP.items()}


class CommonSkeletonTracker:
    """세션 동안 상태(직전 신뢰값)를 유지하며 MediaPipe 프레임을 Common Skeleton으로 변환."""

    def __init__(self, min_visibility: float = 0.5):
        self.min_visibility = min_visibility
        self.last_good = np.zeros((len(COMMON_JOINT_NAMES), 2))
        self.has_good = np.zeros(len(COMMON_JOINT_NAMES), dtype=bool)

    def update(self, landmarks_px: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """반환: (common 18x2 좌표(결측은 freeze 처리됨), frozen_mask(18,) bool, mean_confidence)."""
        raw = np.zeros((len(COMMON_JOINT_NAMES), 2))
        conf = np.zeros(len(COMMON_JOINT_NAMES))

        for i, name in enumerate(COMMON_JOINT_NAMES):
            if name == "Hip":
                idx_l, idx_r = _MP_IDX["LEFT_HIP"], _MP_IDX["RIGHT_HIP"]
                raw[i] = (landmarks_px[idx_l] + landmarks_px[idx_r]) / 2.0
                conf[i] = min(visibility[idx_l], visibility[idx_r])
            elif name == "Neck":
                idx_l, idx_r = _MP_IDX["LEFT_SHOULDER"], _MP_IDX["RIGHT_SHOULDER"]
                raw[i] = (landmarks_px[idx_l] + landmarks_px[idx_r]) / 2.0
                conf[i] = min(visibility[idx_l], visibility[idx_r])
            else:
                idx = _DIRECT_MP_IDX[name]
                raw[i] = landmarks_px[idx]
                conf[i] = visibility[idx]

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
                    out[i] = raw[i]  # 아직 신뢰값 없었으면 저신뢰라도 사용 (0으로 두는 것보다 나음)

        mean_conf = float(np.mean(conf))
        return out, frozen, mean_conf
