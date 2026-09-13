"""우측 패널용 정상 레퍼런스 스켈레톤 트랙.

**정상 배속(원본 AI Hub 30fps 캡처 속도) 재생을 "동작의 기준 속도"로 삼는다.**
사용자의 실시간 동작과 화면상으로 동기화(추적/보폭 조정)하지 않고, 레퍼런스는
독립적으로 계속 반복 재생된다 — 마치 트레이너 시범 영상처럼.

사용자 동작이 이 기준과 얼마나 일치하는지 판정하는 것은 화면 동기화가 아니라
**DTW(phase-aware weighted DTW, `ai_trainer.online_dtw.OnlineSquatSession`)**의
몫이다. DTW는 두 시퀀스 사이의 프레임 단위 타이밍 차이(사용자가 조금 빠르거나
느리게 움직이는 것)를 정렬로 흡수해 비교하므로, 화면을 사용자에 맞춰 억지로
늘였다 줄였다 할 필요가 없다 — 그건 오히려 "레퍼런스가 내 속도에 맞춰 따라온다"는
잘못된 인상을 주고, 정작 사용자가 기준 속도보다 느린지 빠른지 스스로 느끼기
어렵게 만든다.

(참고: 이전에는 사용자의 pelvis 높이에 실시간으로 커서를 맞추는 방식이었으나,
이는 "레퍼런스가 기준 속도를 보여준다"는 목적과 맞지 않아 이 방식으로 교체했다.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..reference_levels import DIFFICULTY_LEVELS, NORMAL_CLASS, validate_difficulty_level

ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = ROOT / "output" / "reference_db"

REFERENCE_FPS = 30.0  # AI Hub 원본 캡처 fps (검증 완료, claude.md 7장) — 재생 타이머 주기로 사용


def _manifest_entries() -> list[dict]:
    return json.loads((DB_DIR / "manifest.json").read_text(encoding="utf-8"))["entries"]


def _medoid_rank(entry: dict) -> int:
    """v2의 명시적 rank를 우선하고, 구 manifest ID 형식은 fallback으로 지원한다."""
    if "medoid_rank" in entry:
        return int(entry["medoid_rank"])
    try:
        return int(entry["medoid_id"].split("_")[1])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Reference DB entry의 medoid rank를 해석할 수 없습니다: {entry!r}") from error


class ReferenceTrack:
    def __init__(
        self,
        class_label: str = NORMAL_CLASS,
        medoid_rank: int = 0,
        tier: str = "ground_truth",
        difficulty_level: str | None = None,
    ):
        level = (
            validate_difficulty_level(difficulty_level)
            if difficulty_level is not None
            else None
        )
        manifest = _manifest_entries()

        entry = None
        for e in manifest:
            if e.get("class_label") != class_label or e.get("tier") != tier:
                continue
            if level is not None and e.get("difficulty_level") != level:
                continue
            if _medoid_rank(e) == medoid_rank:
                entry = e
                break
        if entry is None:
            level_text = f", 난이도={level}" if level is not None else ""
            raise ValueError(
                f"Reference DB에 {class_label} medoid #{medoid_rank} "
                f"({tier}{level_text})가 없습니다."
            )

        with np.load(DB_DIR / "sequences.npz") as arrays:
            # NpzFile context가 닫힌 뒤에도 안전하게 사용할 수 있도록 소유 배열로 복사한다.
            self.coords = arrays[entry["array_key"]].copy()
        self.bounds: dict[str, tuple[int, int]] = {p: tuple(v) for p, v in entry["phase_boundaries"].items()}
        self.meta = entry
        self._cursor = 0

    def lateral_vertical(self, t: int) -> np.ndarray:
        t = max(0, min(t, self.coords.shape[0] - 1))
        return self.coords[t][:, [0, 1]]

    def step(self) -> np.ndarray:
        """정상 배속 재생: 한 프레임 진행하고(끝에 도달하면 처음부터 반복) 좌표를 반환한다.

        REFERENCE_FPS 주기의 타이머에서 호출되도록 설계 — 사용자 입력/카메라 fps와
        완전히 무관하게 항상 같은 실제 속도로 흘러간다.
        """
        xy = self.lateral_vertical(self._cursor)
        self._cursor = (self._cursor + 1) % self.coords.shape[0]
        return xy

    @property
    def current_frame(self) -> int:
        return self._cursor

    def phase_bounds_by_index(self, phase_index: int) -> tuple[int, int]:
        """저장된 phase 순서(준비·하강·최저점·상승·완료)로 구간을 반환한다."""
        values = list(self.bounds.values())
        if phase_index < 0 or phase_index >= len(values):
            return (0, self.coords.shape[0])
        start, end = values[phase_index]
        return max(0, int(start)), min(self.coords.shape[0], int(end))

    def phase_at(self, t: int | None = None) -> str:
        t = self._cursor if t is None else t
        for phase, (s, e) in self.bounds.items():
            if s <= t < e:
                return phase
        return "-"


def list_available(
    tier: str = "ground_truth",
    difficulty_level: str | None = None,
) -> list[tuple[str, int]]:
    """사용 가능한 ``(class_label, medoid_rank)`` 목록을 반환한다."""
    level = (
        validate_difficulty_level(difficulty_level)
        if difficulty_level is not None
        else None
    )
    available = set()
    for entry in _manifest_entries():
        if entry.get("tier") != tier:
            continue
        if level is not None and entry.get("difficulty_level") != level:
            continue
        available.add((entry["class_label"], _medoid_rank(entry)))
    return sorted(available)


def create_normal_tracks(tier: str = "ground_truth") -> dict[str, ReferenceTrack]:
    """초급·중급·고급의 정상 rank-0 트랙을 모두 로드한다."""
    return {
        level: ReferenceTrack(
            class_label=NORMAL_CLASS,
            medoid_rank=0,
            tier=tier,
            difficulty_level=level,
        )
        for level in DIFFICULTY_LEVELS
    }
