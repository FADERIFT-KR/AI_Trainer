"""ms.choe의 `live_pose.PoseObservation`(MediaPipe 33관절) -> 이 브랜치의
Common Skeleton(18노드) 변환 + 저신뢰 관절 freeze 처리.

`ai_trainer.pose_estimator`/`mediapipe_mapping`(feature/dtw-pipeline에서 제거됨)과
같은 역할이지만, ms.choe의 `PoseObservation.image_landmarks`
((33,4) float32 array: 정규화 x,y, z, visibility) 형식에 맞춰 새로 작성했다.
"""
from __future__ import annotations

import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES

from .one_euro_filter import OneEuroFilter

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
        # 관절별 픽셀좌표 프레임간 지터 저감(1€ Filter) — heel/ankle처럼 MediaPipe
        # 신뢰도가 낮아 흔들리기 쉬운 관절이 정지 구간에서도 떨려 보이고, 그 떨림이
        # 하강/상승 중 DTW feature(heel_height/pelvis_trajectory)에 노이즈로 섞여 들어가
        # 오탐을 유발하던 문제(실사용 확인) 완화용. one_euro_filter.py 참고.
        self._smoother = OneEuroFilter(n_points=len(COMMON_JOINT_NAMES), n_dims=2)

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

        good_mask = conf >= self.min_visibility
        raw = self._smoother(raw, good_mask)

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

        return out, frozen, float(np.mean(conf))


class CommonSkeleton3DBridge:
    """MediaPipe `world_landmarks`(BlazePose 자체 3D, 미터 단위, 대략 Hip 중심) ->
    Common Skeleton(18노드) 3D 변환.

    이 프로젝트가 학습한 소규모 2D->3D lifting 모델(TemporalLiftingNet, AI Hub camera1
    단일 카메라 데이터 12만개 파라미터로만 학습)은 실제 아이폰 촬영 영상에서 학습 분포
    밖의 체형/팔자세/화각을 만나면 스쿼트 깊이를 심하게 과소평가하는 것이 확인됐다
    (2026-08-28, 최저점 골반-발목 수직거리: lifting 모델 13%감소 vs 실제로는 훨씬 큼).
    반면 MediaPipe 자체 world_landmarks(Google이 훨씬 크고 다양한 데이터로 학습)는 같은
    프레임에서 35% 감소로 실제 깊은 스쿼트와 일치했다. 그래서 실시간 파이프라인은 자체
    lifting 모델 대신 이미 매 프레임 함께 나오는 이 3D를 그대로 쓴다(추론 1회 재사용,
    지연 없음 — lifting 모델의 4프레임 지연 윈도우도 필요 없어짐).

    AI Hub 데이터는 CSV/JSON만 쓴다는 프로젝트 방침(claude.md)상 원본 영상에 MediaPipe를
    돌릴 수 없어 레퍼런스 DB 쪽은 그대로 CSV 기반(ground_truth/operational)을 유지한다
    — 이 브릿지는 실시간 사용자 입력 쪽에만 쓰이고, 비교 대상은 ground_truth tier(8카메라
    삼각측량 실측 3D)로 맞춘다(둘 다 "실제 3D"라는 도메인이 이제는 lifting 모델을 거치지
    않고도 서로 맞아떨어짐).

    2D 브릿지(CommonSkeletonBridge)와 동일한 confidence-freeze + 지터 저감(1€ Filter)
    패턴을 그대로 적용한다.
    """

    def __init__(self, min_visibility: float = 0.5):
        self.min_visibility = min_visibility
        self.last_good = np.zeros((len(COMMON_JOINT_NAMES), 3))
        self.has_good = np.zeros(len(COMMON_JOINT_NAMES), dtype=bool)
        # OneEuroFilter의 기본 min_cutoff/beta(0.8/0.02)는 CommonSkeletonBridge(2D 픽셀
        # 좌표, 프레임간 이동량이 보통 수~수백 px)를 기준으로 잡은 값이다. world_landmarks는
        # 미터 단위(발목 기준 실측 프레임간 이동량 최대 ~1m/s, 정지 시 ~0.05m/s)라 그대로
        # 재사용하면 beta*dx 항이 min_cutoff에 비해 거의 0(예: 1.0*0.02=0.02)이 되어 버려
        # "빠르게 움직일 때 스무딩을 푼다"는 필터의 핵심 동작이 사실상 꺼진 채로 항상 강하게
        # 스무딩만 걸렸다 — 빠른 하강(0.4m/1초) 시뮬레이션 지연이 beta=0.02일 땐 5.6프레임
        # (~190ms)이나 됐는데, beta=12.0으로 미터 단위에 맞게 재조정하니 1프레임 이하로
        # 줄어드는 것으로 확인됨(2026-09-01). 이 지연이 골반 높이/속도 신호를 왜곡시켜
        # REP 종료 판정이 늦어지거나(최종 프레임에서 상승이 안 끝남) phase 상태기계가
        # 노이즈성 속도 값에 오작동해 REP가 잘못 세어지고 그 어중간한 프레임이 오류로
        # 오분류되는 문제(실사용 확인: "카운팅이 이상함", "하방오류가 계속 뜸")의 원인으로
        # 추정된다.
        self._smoother = OneEuroFilter(n_points=len(COMMON_JOINT_NAMES), n_dims=3, min_cutoff=0.8, beta=12.0)

    def update(self, world_landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """world_landmarks: (33,4) [x,y,z,visibility], 미터 단위 실좌표.

        반환: (common 18x3 3D좌표(결측은 freeze), frozen_mask(18,), mean_confidence)
        """
        pos_all = world_landmarks[:, :3]
        vis_all = world_landmarks[:, 3]

        raw = np.zeros((len(COMMON_JOINT_NAMES), 3))
        conf = np.zeros(len(COMMON_JOINT_NAMES))
        for i, name in enumerate(COMMON_JOINT_NAMES):
            if name == "Hip":
                l, r = _MP_INDEX["LEFT_HIP"], _MP_INDEX["RIGHT_HIP"]
                raw[i] = (pos_all[l] + pos_all[r]) / 2.0
                conf[i] = min(vis_all[l], vis_all[r])
            elif name == "Neck":
                l, r = _MP_INDEX["LEFT_SHOULDER"], _MP_INDEX["RIGHT_SHOULDER"]
                raw[i] = (pos_all[l] + pos_all[r]) / 2.0
                conf[i] = min(vis_all[l], vis_all[r])
            else:
                idx = _DIRECT_MP_IDX[name]
                raw[i] = pos_all[idx]
                conf[i] = vis_all[idx]

        good_mask = conf >= self.min_visibility
        raw = self._smoother(raw, good_mask)

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

        # MediaPipe world_landmarks는 대략 Hip 중심이지만, online_dtw._process_3d_frame이
        # 기대하는 "정확히 Hip=원점" 계약(lifting 모델 경로와 동일)을 보장하기 위해 우리
        # 정의(LHip/RHip 평균)로 명시적으로 재중심화한다.
        out = out - out[COMMON_JOINT_NAMES.index("Hip")]

        return out, frozen, float(np.mean(conf))
