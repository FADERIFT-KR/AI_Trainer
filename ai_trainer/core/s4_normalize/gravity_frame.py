"""AI Hub·MediaPipe 공통 중력 기준 좌표계.

출력 좌표계 (오른손 좌표계, 단위 m):
    x = 피험자의 오른쪽, y = 피험자의 정면, z = 위(중력 반대)
    원점 = 프레임마다 Pelvis(좌우 고관절 중점)

원점을 프레임마다 Pelvis에 두는 것은 MediaPipe world landmarks가 원래 그런 형태이기
때문이다(원점 = 고관절 중점). 따라서 Pelvis의 절대 높이는 쓸 수 없고, 판정 지표는
한 프레임 안의 상대 위치·각도로만 계산해야 한다.

축은 시퀀스의 선 자세 구간에서 한 번만 정하고 전체 프레임에 같은 회전을 적용한다.
프레임마다 축을 다시 잡으면 스쿼트 중 상체 기울기 같은 판정 신호가 사라진다.

- 선 자세 구간: 좌우 무릎각이 시퀀스 최댓값에서 ``tolerance_deg`` 이내인 프레임.
  무릎각은 회전과 무관하므로 축을 정하기 전에 고를 수 있다.
- 위쪽(z): 선 자세의 발목 중점 → 어깨 중점 방향 평균. AI Hub 361개 시퀀스에서 실제
  수직축과의 오차가 중앙값 1.4°, 95% 3.3°로, 몸통 축(고관절→어깨, 중앙값 2.4°)이나
  발 평면 법선(최대 18.5°)보다 정확했다.
- 오른쪽(x): RHip - LHip에서 위쪽 성분을 뺀 방향. 정면(y) = z × x.
- 손잡이 검사: 정면(y)과 발끝 방향(뒤꿈치→발끝)이 반대면 좌우가 뒤바뀐 입력(거울
  좌표계 또는 좌우 라벨 반전)이므로 ``ValueError``를 낸다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _mid(seq: np.ndarray, left: str, right: str) -> np.ndarray:
    return (seq[..., _IDX[left], :] + seq[..., _IDX[right], :]) / 2.0


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """b를 꼭짓점으로 하는 a-b-c 내각(도). 완전히 펴지면 180."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def knee_angles_deg(seq: np.ndarray) -> np.ndarray:
    """(T, 18, 3) → (T,) 좌우 평균 무릎 내각."""
    left = joint_angle_deg(seq[:, _IDX["LHip"]], seq[:, _IDX["LKnee"]], seq[:, _IDX["LAnkle"]])
    right = joint_angle_deg(seq[:, _IDX["RHip"]], seq[:, _IDX["RKnee"]], seq[:, _IDX["RAnkle"]])
    return (left + right) / 2.0


def standing_mask(seq: np.ndarray, tolerance_deg: float = 5.0) -> np.ndarray:
    """무릎이 가장 펴진 프레임들(선 자세)을 True로 표시한다."""
    knee = knee_angles_deg(seq)
    return knee >= np.nanmax(knee) - tolerance_deg


@dataclass(frozen=True)
class GravityFrame:
    rotation: np.ndarray  # (3, 3), 행 = 입력 좌표계에서 본 [오른쪽, 정면, 위] 단위벡터
    standing: np.ndarray  # (T,) bool, 축을 정하는 데 쓴 프레임
    handedness: float  # 정면축과 발끝 방향의 코사인. 양수여야 정상

    @property
    def up(self) -> np.ndarray:
        return self.rotation[2]


def fit_gravity_frame(
    seq: np.ndarray, up: np.ndarray | None = None, tolerance_deg: float = 5.0
) -> GravityFrame:
    """(T, 18, 3) Common Skeleton 시퀀스에서 중력 기준 축을 정한다.

    ``up``을 주면(예: AI Hub의 실제 수직축 (0, 0, 1)) 추정 대신 그 값을 쓴다. 소스 간
    비교에서는 양쪽에 같은 추정 방법을 쓰는 것이 편향을 맞추는 데 유리하다.
    """
    if seq.ndim != 3 or seq.shape[1:] != (len(COMMON_JOINT_NAMES), 3):
        raise ValueError(f"expected (T, 18, 3), got {seq.shape}")
    stand = standing_mask(seq, tolerance_deg)
    if up is None:
        up_vec = _unit((_mid(seq, "LShoulder", "RShoulder") - _mid(seq, "LAnkle", "RAnkle"))[stand].mean(0))
    else:
        up_vec = _unit(np.asarray(up, dtype=np.float64))

    right = (seq[:, _IDX["RHip"]] - seq[:, _IDX["LHip"]])[stand].mean(0)
    right = _unit(right - (right @ up_vec) * up_vec)
    forward = np.cross(up_vec, right)

    toe = (_mid(seq, "LBigToe", "RBigToe") - _mid(seq, "LHeel", "RHeel"))[stand].mean(0)
    toe = _unit(toe - (toe @ up_vec) * up_vec)
    handedness = float(forward @ toe)
    if handedness <= 0.0:
        raise ValueError(
            f"정면축과 발끝 방향이 반대입니다(cos={handedness:.2f}). 좌우가 뒤바뀐 좌표계이거나 "
            "좌우 라벨이 반전된 입력입니다."
        )
    return GravityFrame(np.stack([right, forward, up_vec]), stand, handedness)


def to_gravity_frame(seq: np.ndarray, frame: GravityFrame) -> np.ndarray:
    """(T, 18, 3) → 프레임별 Pelvis 원점, [오른쪽, 정면, 위] 축 좌표 (T, 18, 3)."""
    centered = seq - seq[:, _IDX["Hip"] : _IDX["Hip"] + 1, :]
    return centered @ frame.rotation.T


def align_sequence(
    seq: np.ndarray, up: np.ndarray | None = None, tolerance_deg: float = 5.0
) -> tuple[np.ndarray, GravityFrame]:
    """``fit_gravity_frame`` + ``to_gravity_frame``."""
    frame = fit_gravity_frame(seq, up=up, tolerance_deg=tolerance_deg)
    return to_gravity_frame(seq, frame), frame
