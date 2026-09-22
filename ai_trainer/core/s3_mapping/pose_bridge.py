"""MediaPipe `PoseObservation`(33관절) -> Common Skeleton(18노드) 변환 + 저신뢰 관절 freeze.

시간축 필터(One-Euro, 스파이크 가드, 다리길이 안정화)와 표시용 기하 보정은 제거했다.
MediaPipe 추정값을 그대로 쓰고, 이 모듈이 하는 일은 관절 매핑과 저신뢰 관절 freeze뿐이다.
"""
from __future__ import annotations

import numpy as np

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES

# MediaPipe PoseLandmark 인덱스 (Task API, 33점 — live_pose.render.POSE_CONNECTIONS와 동일 토폴로지)
_MP_INDEX = {
    "NOSE": 0, "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7, "RIGHT_EAR": 8, "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12, "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16, "LEFT_PINKY": 17, "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19, "RIGHT_INDEX": 20, "LEFT_THUMB": 21, "RIGHT_THUMB": 22,
    "LEFT_HIP": 23, "RIGHT_HIP": 24, "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28, "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}

_DIRECT_MAP = {
    "Nose": "NOSE",
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


def _map_to_common(points: np.ndarray, visibility: np.ndarray, n_dims: int):
    """33관절 -> Common 18관절. Hip/Neck은 좌우 평균, 신뢰도는 둘 중 낮은 값."""
    raw = np.zeros((len(COMMON_JOINT_NAMES), n_dims))
    conf = np.zeros(len(COMMON_JOINT_NAMES))
    for i, name in enumerate(COMMON_JOINT_NAMES):
        if name in ("Hip", "Neck"):
            left, right = ("LEFT_HIP", "RIGHT_HIP") if name == "Hip" else ("LEFT_SHOULDER", "RIGHT_SHOULDER")
            l, r = _MP_INDEX[left], _MP_INDEX[right]
            raw[i] = (points[l] + points[r]) / 2.0
            conf[i] = min(visibility[l], visibility[r])
        else:
            idx = _DIRECT_MP_IDX[name]
            raw[i] = points[idx]
            conf[i] = visibility[idx]
    return raw, conf


class _FreezeMixin:
    """저신뢰 관절은 마지막으로 잘 잡힌 좌표를 유지(freeze)한다."""

    def _apply_freeze(self, raw: np.ndarray, good_mask: np.ndarray):
        frozen = np.zeros(len(COMMON_JOINT_NAMES), dtype=bool)
        out = self.last_good.copy()
        for i in range(len(COMMON_JOINT_NAMES)):
            if good_mask[i]:
                out[i] = raw[i]
                self.last_good[i] = raw[i]
                self.has_good[i] = True
            else:
                frozen[i] = True
                if not self.has_good[i]:
                    out[i] = raw[i]
        return out, frozen


class CommonSkeletonBridge(_FreezeMixin):
    """image_landmarks(정규화 2D) -> Common Skeleton 18관절 픽셀 좌표."""

    def __init__(self, min_visibility: float = 0.5):
        self.min_visibility = min_visibility
        self.last_good = np.zeros((len(COMMON_JOINT_NAMES), 2))
        self.has_good = np.zeros(len(COMMON_JOINT_NAMES), dtype=bool)

    def update(self, image_landmarks: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, float]:
        """image_landmarks: (33,4) [x,y,z,visibility], x/y는 [0,1] 정규화.

        반환: (common 18x2 픽셀좌표(결측은 freeze), frozen_mask(18,), mean_confidence)
        """
        px_all = image_landmarks[:, :2] * np.array([width, height])
        raw, conf = _map_to_common(px_all, image_landmarks[:, 3], n_dims=2)
        out, frozen = self._apply_freeze(raw, conf >= self.min_visibility)
        return out, frozen, float(np.mean(conf))


class CommonSkeleton3DBridge(_FreezeMixin):
    """MediaPipe `world_landmarks`(미터 단위, 대략 Hip 중심) -> Common Skeleton 18관절 3D.

    이 프로젝트가 학습한 소규모 2D->3D lifting 모델(TemporalLiftingNet, AI Hub camera1
    단일 카메라 데이터 12만개 파라미터로만 학습)은 실제 촬영 영상에서 학습 분포 밖의
    체형/팔자세/화각을 만나면 스쿼트 깊이를 심하게 과소평가하는 것이 확인됐다
    (2026-08-28, 최저점 골반-발목 수직거리: lifting 모델 13%감소 vs 실제로는 훨씬 큼).
    반면 MediaPipe 자체 world_landmarks(Google이 훨씬 크고 다양한 데이터로 학습)는 같은
    프레임에서 35% 감소로 실제 깊은 스쿼트와 일치했다. 그래서 실시간 파이프라인은 자체
    lifting 모델 대신 이미 매 프레임 함께 나오는 이 3D를 그대로 쓴다.

    AI Hub 데이터는 CSV/JSON만 쓴다는 프로젝트 방침(claude.md)상 원본 영상에 MediaPipe를
    돌릴 수 없어 레퍼런스 DB 쪽은 CSV 기반(ground_truth/operational)을 유지한다 — 이
    브릿지는 실시간 사용자 입력 쪽에만 쓰이고, 비교 대상은 ground_truth tier(8카메라
    삼각측량 실측 3D)로 맞춘다.
    """

    def __init__(self, min_visibility: float = 0.5):
        self.min_visibility = min_visibility
        self.last_good = np.zeros((len(COMMON_JOINT_NAMES), 3))
        self.has_good = np.zeros(len(COMMON_JOINT_NAMES), dtype=bool)

    def update(self, world_landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """world_landmarks: (33,4) [x,y,z,visibility], 미터 단위 실좌표.

        반환: (common 18x3 3D좌표(결측은 freeze), frozen_mask(18,), mean_confidence)
        """
        raw, conf = _map_to_common(world_landmarks[:, :3], world_landmarks[:, 3], n_dims=3)
        good_mask = (conf >= self.min_visibility) & np.isfinite(raw).all(axis=1)
        out, frozen = self._apply_freeze(raw, good_mask)

        # 파생 pelvis는 좌우 hip의 중점과 정확히 일치해야 한다. 따로 두면 두 hip이
        # 일관되지 않은 원점을 중심으로 흔들리는 것처럼 보인다.
        pelvis_index = COMMON_JOINT_NAMES.index("Hip")
        left_hip_index = COMMON_JOINT_NAMES.index("LHip")
        right_hip_index = COMMON_JOINT_NAMES.index("RHip")
        if self.has_good[left_hip_index] and self.has_good[right_hip_index]:
            out[pelvis_index] = (out[left_hip_index] + out[right_hip_index]) / 2.0
            self.last_good[pelvis_index] = out[pelvis_index]

        # MediaPipe world_landmarks는 대략 Hip 중심이지만, online_dtw._process_3d_frame이
        # 기대하는 "정확히 Hip=원점" 계약을 보장하기 위해 명시적으로 재중심화한다.
        out = out - out[pelvis_index]
        return out, frozen, float(np.mean(conf))


__all__ = ["CommonSkeletonBridge", "CommonSkeleton3DBridge"]
