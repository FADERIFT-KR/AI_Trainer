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


# "정상" reference와의 절대 거리가 이 유사도(%) 이상이면, 다른 오류 클래스가
# DTW distance상 근소하게 더 가깝다는 이유만으로 특정 오류로 확정 판정하지 않고
# "자세 양호"로 처리한다. 클래스 간 상대 비교(nearest-neighbor)만으로 판정하면
# 상승 구간처럼 자세가 살짝만 틀어져도(=정상 레퍼런스에서 크게 벗어나지 않았는데도)
# 우연히 어느 오류 패턴에 조금 더 가까워 보이는 경우 바로 오류로 뜨는 문제가 있었다
# (실사용 확인됨). "정상 CSV 좌표에서 많이 벗어난 경우만 오류로 표시" 요구사항 반영.
PASS_SCORE_THRESHOLD = 35.0
