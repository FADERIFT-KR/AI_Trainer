"""AI Hub 26관절 → Common Skeleton(18노드) 매핑.

claude.md 4장의 Common Skeleton Mapping 표를 코드로 구현한다. AI Hub CSV는
Pelvis/Neck에 해당하는 ``Hip``/``Neck`` 컬럼을 이미 직접 제공하므로, AI Hub
쪽에서는 좌우 평균으로 파생시키지 않고 제공값을 그대로 사용한다(MediaPipe
쪽에서만 좌우 평균으로 파생시킨다 — 실시간 파이프라인 구현 시 반영).

``Head``, ``LSmallToe``, ``RSmallToe``는 MediaPipe에 대응 랜드마크가 없어 제외한다.
"""
from __future__ import annotations

import numpy as np

from .aihub_zip import JOINT_NAMES

# 18노드 순서: 좌우 8쌍 + Pelvis(Hip) + Neck
COMMON_JOINT_NAMES: list[str] = [
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LWrist", "RWrist",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
    "LHeel", "RHeel",
    "LBigToe", "RBigToe",
    "Hip",   # Pelvis
    "Neck",
]

assert len(COMMON_JOINT_NAMES) == 18

# AI Hub 26관절 인덱스 배열 중 Common Skeleton 18노드에 해당하는 위치
_AIHUB_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}
COMMON_FROM_AIHUB_IDX: np.ndarray = np.array(
    [_AIHUB_INDEX[name] for name in COMMON_JOINT_NAMES], dtype=np.int64
)

PELVIS_IDX = COMMON_JOINT_NAMES.index("Hip")
NECK_IDX = COMMON_JOINT_NAMES.index("Neck")
L_HIP_IDX, R_HIP_IDX = COMMON_JOINT_NAMES.index("LHip"), COMMON_JOINT_NAMES.index("RHip")


def to_common_skeleton(coords_26: np.ndarray) -> np.ndarray:
    """(..., 26, D) AI Hub 좌표 배열을 (..., 18, D) Common Skeleton으로 축소한다."""
    return coords_26[..., COMMON_FROM_AIHUB_IDX, :]


# 시각화/디버깅용 뼈대 연결 (18노드 기준, 얼굴 관절 없음)
_COMMON_BONES_NAMES: list[tuple[str, str]] = [
    ("Neck", "LShoulder"), ("Neck", "RShoulder"),
    ("Neck", "Hip"),
    ("Hip", "LHip"), ("Hip", "RHip"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"),
    ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"),
]
_COMMON_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
COMMON_BONE_INDEX_PAIRS: list[tuple[int, int]] = [
    (_COMMON_IDX[a], _COMMON_IDX[b]) for a, b in _COMMON_BONES_NAMES
]


def _color(a: str, b: str) -> tuple[int, int, int]:
    if a.startswith("L") or b.startswith("L"):
        return (255, 120, 0)
    if a.startswith("R") or b.startswith("R"):
        return (0, 60, 255)
    return (230, 230, 230)


COMMON_BONE_COLORS_BGR: list[tuple[int, int, int]] = [
    _color(a, b) for a, b in _COMMON_BONES_NAMES
]
