"""우측 패널용 정상 레퍼런스 스켈레톤 트랙.

Reference DB(이 브랜치가 AI Hub CSV/JSON으로 구축한 것)에서 한 medoid 시퀀스를
불러와, 현재 사용자의 phase에 맞춰 재생 위치를 진행시키는 상태를 관리한다.

동기화 방식(v1, 단순화): 프레임마다 "현재 phase가 바뀌면 그 phase 레퍼런스 구간의
시작으로 이동, 같은 phase가 유지되면 진행(구간 끝에서는 유지)".
DTW 정렬 기반의 정밀한 진행률 동기화는 다음 단계 개선 대상이다(README 참고).

레퍼런스 데이터(AI Hub)는 30fps로 캡처되었다(annotation.json의 start_time/end_time
대비 start_frame/end_frame 역산으로 검증 완료). 웹캠도 `CameraConfig.requested_fps=30`
으로 30fps를 목표로 하지만, MediaPipe 추론 부하 등으로 실제 처리 속도가 30fps에 못
미칠 수 있다. 그래서 `advance()`는 "틱마다 무조건 1프레임"이 아니라, 실측 fps를 받아
`live_fps/30`만큼 누적 진행시켜 **실제 웹캠 fps가 30에서 벗어나도 레퍼런스가 항상
실제 시간(wall-clock) 기준으로는 30fps 소스와 같은 속도로 재생**되도록 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = ROOT / "output" / "reference_db"
REFERENCE_FPS = 30.0  # AI Hub 원본 캡처 fps (검증 완료, claude.md 7장 참고)

# OnlineSquatSession 내부 상태(영문) -> phase_boundaries 키(국문)
STATE_TO_PHASE = {"prep": "준비", "descend": "하강", "bottom": "최저점", "ascend": "상승"}
PHASE_ORDER = ["준비", "하강", "최저점", "상승", "종료"]


class ReferenceTrack:
    def __init__(self, class_label: str = "정상", medoid_rank: int = 0, tier: str = "operational"):
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

        self._current_phase = "준비"
        self._cursor = self.bounds.get("준비", (0, 1))[0]
        self._frac_accum = 0.0  # 1프레임 미만 진행분 누적(실측 fps가 30이 아닐 때 보정용)

    def lateral_vertical(self, t: int) -> np.ndarray:
        t = max(0, min(t, self.coords.shape[0] - 1))
        return self.coords[t][:, [0, 1]]

    def advance(self, live_state: str | None, live_fps: float | None = None) -> np.ndarray:
        """live_state: OnlineSquatSession의 영문 상태('prep'/'descend'/'bottom'/'ascend') 또는 None.
        live_fps: 이번 틱의 실측 웹캠 처리 fps. 30에서 벗어나면 그 비율만큼 진행 속도를 보정해
        레퍼런스(30fps 원본)가 항상 실제 시간 기준으로 올바른 속도로 재생되게 한다.

        매 호출마다 진행한 레퍼런스 lateral-vertical (18,2) 좌표를 반환한다.
        """
        phase = STATE_TO_PHASE.get(live_state, "준비") if live_state else "준비"
        if phase != self._current_phase:
            self._current_phase = phase
            self._cursor = self.bounds.get(phase, (0, 1))[0]
            self._frac_accum = 0.0
        else:
            s, e = self.bounds.get(phase, (0, self.coords.shape[0]))
            step = (live_fps / REFERENCE_FPS) if live_fps and live_fps > 0 else 1.0
            self._frac_accum += step
            while self._frac_accum >= 1.0 and self._cursor < e - 1:
                self._cursor += 1
                self._frac_accum -= 1.0
        return self.lateral_vertical(self._cursor)

    @property
    def current_phase(self) -> str:
        return self._current_phase


def list_available(tier: str = "operational") -> list[tuple[str, int]]:
    manifest = json.loads((DB_DIR / "manifest.json").read_text(encoding="utf-8"))["entries"]
    return sorted(
        {(e["class_label"], int(e["medoid_id"].split("_")[1])) for e in manifest if e["tier"] == tier}
    )
