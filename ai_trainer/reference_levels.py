"""Reference DB의 난이도/자세 클래스 공통 정의."""
from __future__ import annotations

DIFFICULTY_LEVELS: tuple[str, ...] = ("초급", "중급", "고급")
REFERENCE_CLASSES: tuple[str, ...] = (
    "정상",
    "발뒤꿈치오류",
    "엉덩이하방오류",
    "고관절오류",
)
NORMAL_CLASS = "정상"


def validate_difficulty_level(level: str) -> str:
    """지원하는 난이도인지 검증하고 원래 값을 반환한다."""
    if level not in DIFFICULTY_LEVELS:
        allowed = ", ".join(DIFFICULTY_LEVELS)
        raise ValueError(f"알 수 없는 난이도: {level!r} (지원: {allowed})")
    return level


__all__ = [
    "DIFFICULTY_LEVELS",
    "REFERENCE_CLASSES",
    "NORMAL_CLASS",
    "validate_difficulty_level",
]
