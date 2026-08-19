"""DTW distance -> 0~100 자세 점수 변환.

임의 고정 선형식 대신, 정상 reference distance 분포(하위 percentile)와
오류 시퀀스 distance 분포(중앙값)를 기준으로 두 앵커를 잡는다.
"""
from __future__ import annotations

import numpy as np


def fit_score_calibration(normal_distances: np.ndarray, error_distances: np.ndarray) -> dict:
    lo = float(np.percentile(normal_distances, 10))  # 상위(좋은) 앵커: 정상 시퀀스 중에서도 잘한 축에 속함
    hi = float(np.percentile(error_distances, 50))  # 하위(나쁜) 앵커: 오류 시퀀스의 전형적인 거리
    if hi <= lo:
        hi = lo + 1e-6
    return {"lo": lo, "hi": hi}


def distance_to_score(d: float, calib: dict) -> float:
    score = 100.0 * (calib["hi"] - d) / (calib["hi"] - calib["lo"])
    return float(np.clip(score, 0.0, 100.0))
