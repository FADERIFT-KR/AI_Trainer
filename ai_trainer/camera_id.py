"""camera별 2D Skeleton과 3D Skeleton의 orthographic projection을 비교해
정면 카메라 후보를 정량적으로 검증하는 유틸리티.

절차 (사용자 지정 방식):
  1. 3D (T,26,3) 좌표에서 두 축을 골라 orthographic projection (T,26,2)을 만든다.
     후보: X-Y, X-Z, Y-Z
  2. 2D 카메라 좌표와 3D projection 각각을 독립적으로 translation/scale
     normalization 한다.
  3. 공통 관절(26개 전부, 2D/3D CSV가 동일한 관절셋을 공유)을 사용해
     최적 회전(및 반사)까지 허용하는 Procrustes 정합을 수행하고, 정합 후
     잔차(disparity)를 유사도(비유사도) 지표로 사용한다.

Procrustes 정합은 미지의 in-plane 회전/반사를 흡수하므로, "카메라가 정확히
어느 축을 보고 있는가"가 아니라 "이 카메라의 2D 형태가 이 projection 평면의
형태와 얼마나 닮았는가"를 측정한다.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import procrustes

PROJECTIONS: dict[str, tuple[int, int]] = {
    "XY": (0, 1),
    "XZ": (0, 2),
    "YZ": (1, 2),
}


def project(coords_3d_frame: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """(26,3) 3D 프레임 좌표를 지정한 두 축으로 orthographic projection한다 -> (26,2)."""
    return coords_3d_frame[:, axes]


def procrustes_disparity(points_a: np.ndarray, points_b: np.ndarray) -> float | None:
    """두 (J,2) 점집합을 translation/scale 정규화 후 최적 회전으로 정합하고 잔차를 반환한다.

    scipy.spatial.procrustes는 각 입력을 평균 0, Frobenius norm 1로 표준화한 뒤
    최적 직교변환(회전+반사 허용)을 찾으므로 사용자가 지정한 절차와 정확히 일치한다.
    입력이 퇴화(분산 0 등)되어 계산 불가능하면 None을 반환한다.
    """
    try:
        _, _, disparity = procrustes(points_a, points_b)
    except ValueError:
        return None
    return float(disparity)


def sample_frame_indices(start: int, end: int, n_samples: int) -> list[int]:
    """[start, end] 구간에서 균등 간격으로 n_samples개의 정수 프레임 인덱스를 뽑는다."""
    if end <= start:
        return [start]
    idx = np.linspace(start, end, num=min(n_samples, end - start + 1))
    return sorted(set(int(round(i)) for i in idx))


def evaluate_frame(
    coords_2d_frame: np.ndarray, coords_3d_frame: np.ndarray
) -> dict[str, float | None]:
    """한 프레임에 대해 3개 projection 각각의 Procrustes disparity를 계산한다."""
    out: dict[str, float | None] = {}
    for name, axes in PROJECTIONS.items():
        proj = project(coords_3d_frame, axes)
        out[name] = procrustes_disparity(coords_2d_frame, proj)
    return out
