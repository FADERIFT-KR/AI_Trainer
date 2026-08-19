"""pelvis_height/velocity 기반 5-phase 상태기계: 준비 -> 하강 -> 최저점 -> 상승 -> 종료.

`scripts/visualize_phase_features.py`에서 4개 클래스 전부 pelvis_height가 뚜렷한
U자형(준비 근처 최댓값 -> 최저점 -> 준비 수준으로 복귀)을 보임을 확인했으므로,
전역 최솟값(최저점)을 기준으로 좌우로 "속도가 0에 가까운" 구간을 찾아 나누는
단순한 규칙 기반 방식을 사용한다. 시퀀스마다 속도 스케일이 다르므로 임계값은
해당 시퀀스 자체의 |velocity| 최댓값에 대한 비율로 적응적으로 정한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .phase_features import PhaseFeatures

PHASES = ["준비", "하강", "최저점", "상승", "종료"]


@dataclass
class PhaseBoundaries:
    labels: np.ndarray  # (T,) 문자열 라벨
    prep_end: int
    bottom_start: int
    bottom_end: int
    rise_end: int

    def as_dict(self) -> dict:
        return {
            "준비": [0, int(self.prep_end)],
            "하강": [int(self.prep_end), int(self.bottom_start)],
            "최저점": [int(self.bottom_start), int(self.bottom_end)],
            "상승": [int(self.bottom_end), int(self.rise_end)],
            "종료": [int(self.rise_end), int(len(self.labels))],
        }


def segment_phases(features: PhaseFeatures, vel_ratio: float = 0.15) -> PhaseBoundaries:
    height = features.pelvis_height
    t = len(height)

    win = min(11, t if t % 2 == 1 else t - 1)
    win = max(5, win)
    smooth_h = savgol_filter(height, window_length=win, polyorder=2) if t > win else height
    vel = np.gradient(smooth_h)

    bottom_idx = int(np.argmin(smooth_h))
    eps = vel_ratio * (np.max(np.abs(vel)) + 1e-8)

    descending = vel < -eps
    ascending = vel > eps
    near_zero = ~descending & ~ascending

    lo = hi = bottom_idx
    while lo > 0 and near_zero[lo - 1]:
        lo -= 1
    while hi < t - 1 and near_zero[hi + 1]:
        hi += 1

    prep_end = 0
    while prep_end < lo and near_zero[prep_end]:
        prep_end += 1

    end_start = t - 1
    while end_start > hi and near_zero[end_start]:
        end_start -= 1
    end_start += 1

    labels = np.empty(t, dtype=object)
    labels[:prep_end] = "준비"
    labels[prep_end:lo] = "하강"
    labels[lo : hi + 1] = "최저점"
    labels[hi + 1 : end_start] = "상승"
    labels[end_start:] = "종료"

    return PhaseBoundaries(labels=labels, prep_end=prep_end, bottom_start=lo, bottom_end=hi + 1, rise_end=end_start)
