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

ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = ROOT / "output" / "reference_db"

REFERENCE_FPS = 30.0  # AI Hub 원본 캡처 fps (검증 완료, claude.md 7장) — 재생 타이머 주기로 사용


class ReferenceTrack:
    def __init__(self, class_label: str = "정상", medoid_rank: int = 0, tier: str = "ground_truth"):
        manifest = json.loads((DB_DIR / "manifest.json").read_text(encoding="utf-8"))["entries"]
        arrays = np.load(DB_DIR / "sequences.npz")

        entry = None
        for e in manifest:
            rank = int(e["medoid_id"].split("_")[1])
            if e["class_label"] == class_label and rank == medoid_rank and e["tier"] == tier:
                entry = e
                break
        if entry is None:
            raise ValueError(f"Reference DB에 {class_label} medoid #{medoid_rank} ({tier})가 없습니다.")

        self.coords = arrays[entry["array_key"]]  # (T,18,3) 정규화(Hip-center+Scale+Orientation) 완료
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

    def phase_at(self, t: int | None = None) -> str:
        t = self._cursor if t is None else t
        for phase, (s, e) in self.bounds.items():
            if s <= t < e:
                return phase
        return "-"


def list_available(tier: str = "ground_truth") -> list[tuple[str, int]]:
    manifest = json.loads((DB_DIR / "manifest.json").read_text(encoding="utf-8"))["entries"]
    return sorted(
        {(e["class_label"], int(e["medoid_id"].split("_")[1])) for e in manifest if e["tier"] == tier}
    )


DIFFICULTY_LABELS = ["쉬움", "보통", "어려움"]  # 느린 템포 -> 빠른 템포 순


def difficulty_medoid_ranks(
    class_label: str = "정상", tier: str = "ground_truth", n_levels: int = len(DIFFICULTY_LABELS)
) -> list[int]:
    """난이도(쉬움~어려움)에 대응하는 medoid_rank 목록을 반환한다.

    같은 클래스 안에서도 medoid마다 실제 배우가 수행한 하강 속도가 다르다(같은
    "정상" 라벨이어도 phase_boundaries의 하강 구간 길이가 배우별로 최대 2배 이상
    차이남). "난이도"를 이 하강 구간 길이로 정의해, 가장 느린(하강 구간이 긴)
    medoid부터 가장 빠른(짧은) medoid까지 균등 간격으로 n_levels개를 골라
    쉬움->어려움 순으로 정렬해 돌려준다. medoid 구성이 재생성되어도(클러스터링
    재실행 등) 특정 medoid_id를 하드코딩하지 않고 항상 그 시점의 하강 속도
    분포에서 다시 골라내므로 깨지지 않는다.
    """
    manifest = json.loads((DB_DIR / "manifest.json").read_text(encoding="utf-8"))["entries"]
    entries = [e for e in manifest if e["class_label"] == class_label and e["tier"] == tier]
    if not entries:
        raise ValueError(f"Reference DB에 {class_label} ({tier})가 없습니다.")

    def descend_frames(e: dict) -> int:
        s, end = e["phase_boundaries"]["하강"]
        return end - s

    entries.sort(key=descend_frames, reverse=True)  # 느림(긴 하강) -> 빠름(짧은 하강)

    n = min(n_levels, len(entries))
    idxs = np.round(np.linspace(0, len(entries) - 1, n)).astype(int)
    seen: set[int] = set()
    ranks: list[int] = []
    for i in idxs:
        if int(i) in seen:
            continue
        seen.add(int(i))
        ranks.append(int(entries[i]["medoid_id"].split("_")[1]))
    return ranks
