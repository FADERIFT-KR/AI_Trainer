"""AI Hub 26관절 스켈레톤 정의 (관절 인덱스, 연결선 목록, 시각화 색상).

Common Skeleton Mapping(claude.md 계획 문서) 정의 전, 원본 26관절 그대로
시각화할 때 사용하는 뼈대(bone) 연결 정보를 담는다.
"""
from __future__ import annotations

from .aihub_zip import JOINT_NAMES

JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

# (관절A, 관절B) 이름 쌍. 얼굴 → 몸통 → 팔 → 다리 → 발 순서.
BONES: list[tuple[str, str]] = [
    # 얼굴
    ("Nose", "LEye"), ("Nose", "REye"),
    ("LEye", "LEar"), ("REye", "REar"),
    ("Nose", "Head"), ("Head", "Neck"),
    # 몸통
    ("Neck", "LShoulder"), ("Neck", "RShoulder"),
    ("Neck", "Hip"),
    ("Hip", "LHip"), ("Hip", "RHip"),
    # 팔
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    # 다리
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    # 발
    ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"), ("LAnkle", "LSmallToe"),
    ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"), ("RAnkle", "RSmallToe"),
]

BONE_INDEX_PAIRS: list[tuple[int, int]] = [
    (JOINT_INDEX[a], JOINT_INDEX[b]) for a, b in BONES
]


def _bone_color_bgr(a: str, b: str) -> tuple[int, int, int]:
    """OpenCV(BGR) 색상. 좌측=파랑, 우측=빨강, 중앙=흰색."""
    if a.startswith("L") or b.startswith("L"):
        return (255, 120, 0)  # blue-ish
    if a.startswith("R") or b.startswith("R"):
        return (0, 60, 255)  # red-ish
    return (230, 230, 230)  # 중앙선: 밝은 회색


BONE_COLORS_BGR: list[tuple[int, int, int]] = [
    _bone_color_bgr(a, b) for a, b in BONES
]
