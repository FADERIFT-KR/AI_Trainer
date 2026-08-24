"""우측 패널용 정상 레퍼런스 스켈레톤 트랙.

Reference DB(이 브랜치가 AI Hub CSV/JSON으로 구축한 것)에서 한 medoid 시퀀스를
불러와, 현재 사용자의 실제 동작 진행 상태에 맞춰 재생 위치를 진행시키는 상태를 관리한다.

동기화 방식: **pelvis 높이(정규화된 leg_length 단위) 매칭.**
처음에는 "틱마다 1프레임" 식으로 레퍼런스 자체의 원래 속도로 재생했는데, 레퍼런스
phase 구간이 몇 프레임 안 되게 짧다 보니(원본 배우가 실제 사용자보다 빠르게 움직였을
경우가 많음) 레퍼런스가 순식간에 다 재생되고 끝에서 멈춰 기다리는 것처럼 보여
"혼자 너무 빠르게 움직인다"는 문제가 있었다.

지금은 시간(틱 수)이 아니라, 사용자의 **현재 pelvis 높이와 가장 가까운 pelvis 높이를
갖는 레퍼런스 프레임**을 그 phase 구간 안에서 매 틱마다 다시 찾는다. Hip-centered +
Scale + Orientation 정규화를 레퍼런스와 실시간 사용자 양쪽에 동일하게 적용하므로
pelvis 높이가 같은 단위(leg_length)로 비교 가능하다. 이렇게 하면 사용자가 천천히
앉으면 레퍼런스도 천천히, 빨리 앉으면 레퍼런스도 빨리 따라간다 — 실제 움직임 속도와
레퍼런스 재생이 항상 맞물린다.

**튐(jerky) 방지**: 실시간 lifting 출력은 프레임마다 미세하게 흔들리는데(모델 노이즈),
매번 "전체 구간에서 가장 가까운 높이"를 새로 찾으면 그 흔들림 때문에 커서가 앞뒤로
튀어 화면이 뚝뚝 끊겨 보인다. 이를 막기 위해 두 가지를 더한다:
  1) 실시간 pelvis 높이에 지수이동평균(EMA)을 적용해 프레임 간 노이즈를 줄인다.
  2) 매칭 탐색을 "커서 기준 앞으로 몇 프레임"으로만 제한하고, 커서는 절대 뒤로
     가지 않으며 한 틱에 최대 `MAX_STEP_PER_TICK`프레임까지만 이동한다. 이렇게 하면
     항상 완만하게 앞으로만 진행해 부드럽게 보인다(약간의 반응 지연은 감수).

**준비(prep) 상태는 예외적으로 진행시키지 않고 첫 프레임(=대기 자세)에 고정한다.**
사용자가 가만히 서 있어도 관절 추정의 미세한 흔들림 때문에 phase가 계속
재확인되며 커서가 계속 흘러가 "레퍼런스가 혼자 계속 움직이는" 것처럼 보이는 문제가
있었다 — 실제 하강이 감지되기 전까지는 준비 자세 한 프레임만 정지 화면처럼 보여준다.

레퍼런스 데이터(AI Hub)는 30fps로 캡처되었다(annotation.json의 start_time/end_time
대비 start_frame/end_frame 역산으로 검증 완료). pelvis 높이를 못 받는 예외적인
경우를 위한 대비책으로만 fps 기반 진행을 폴백으로 남겨둔다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_trainer.phase_features import extract_phase_features

ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = ROOT / "output" / "reference_db"
REFERENCE_FPS = 30.0  # AI Hub 원본 캡처 fps (검증 완료, claude.md 7장 참고) — fps 폴백에서만 사용
HEIGHT_SMOOTH_ALPHA = 0.3  # 실시간 pelvis 높이 EMA 계수 (낮을수록 더 매끄럽지만 반응은 느려짐)
MAX_STEP_PER_TICK = 4  # 한 틱에 레퍼런스 커서가 최대 이만큼만 전진 (역행/급점프 방지)

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
        self.pelvis_height = extract_phase_features(self.coords).pelvis_height  # (T,) leg_length 단위, 실시간과 같은 척도
        self.bounds: dict[str, tuple[int, int]] = {p: tuple(v) for p, v in entry["phase_boundaries"].items()}
        self.meta = entry

        self._current_phase = "준비"
        self._cursor = self.bounds.get("준비", (0, 1))[0]
        self._frac_accum = 0.0  # fps 폴백용 1프레임 미만 진행분 누적
        self._smoothed_height: float | None = None  # 실시간 pelvis 높이 EMA 상태

    def lateral_vertical(self, t: int) -> np.ndarray:
        t = max(0, min(t, self.coords.shape[0] - 1))
        return self.coords[t][:, [0, 1]]

    def advance(
        self,
        live_state: str | None,
        live_pelvis_height: float | None = None,
        live_fps: float | None = None,
    ) -> np.ndarray:
        """live_state: OnlineSquatSession의 영문 상태('prep'/'descend'/'bottom'/'ascend') 또는 None.
        live_pelvis_height: 사용자의 현재 정규화 pelvis 높이 (동기화의 기준값).
        live_fps: live_pelvis_height가 없을 때만 쓰는 폴백(틱당 진행 속도 보정).

        매 호출마다 진행한 레퍼런스 lateral-vertical (18,2) 좌표를 반환한다.
        """
        phase = STATE_TO_PHASE.get(live_state, "준비") if live_state else "준비"
        if phase != self._current_phase:
            self._current_phase = phase
            s, e = self.bounds.get(phase, (0, 1))
            if phase != "준비" and live_pelvis_height is not None and e > s:
                # phase가 바뀌는 "이 한 틱"만 예외적으로 새 구간 전체에서 즉시 맞는 위치로
                # 점프한다. 하강처럼 높이가 빠르게 변하는 구간은, phase 전환이 감지될
                # 즈음(디바운스 지연) 사용자가 이미 구간 초반보다 훨씬 내려가 있는 경우가
                # 많아서 — 항상 구간 첫 프레임에서 시작해 매 틱 최대 몇 프레임씩만
                # 전진하면 그 차이를 다 못 따라잡고 계속 뒤처진다. 전환 시점 1회만
                # 무제한 탐색으로 스냅하고, 같은 phase가 유지되는 동안(아래 elif)은 기존처럼
                # 부드럽게(역행 없이, 조금씩만) 진행한다.
                seg = self.pelvis_height[s:e]
                offset = int(np.argmin(np.abs(seg - live_pelvis_height)))
                self._cursor = s + offset
            else:
                self._cursor = s
            self._frac_accum = 0.0
            self._smoothed_height = live_pelvis_height  # 새 phase 시작 시 EMA 리셋
        elif phase != "준비":
            s, e = self.bounds.get(phase, (0, self.coords.shape[0]))
            if live_pelvis_height is not None and e > s:
                if self._smoothed_height is None:
                    self._smoothed_height = live_pelvis_height
                else:
                    self._smoothed_height = (
                        HEIGHT_SMOOTH_ALPHA * live_pelvis_height + (1 - HEIGHT_SMOOTH_ALPHA) * self._smoothed_height
                    )
                # 커서보다 앞쪽(진행 방향)의 좁은 구간에서만 최적 매칭 프레임을 찾는다.
                # -> 노이즈로 인한 역행/큰 점프를 원천 차단하고, 항상 조금씩만 앞으로 이동.
                search_start = min(self._cursor, e - 1)
                search_end = min(e, search_start + 1 + MAX_STEP_PER_TICK)
                if search_end > search_start:
                    seg = self.pelvis_height[search_start:search_end]
                    offset = int(np.argmin(np.abs(seg - self._smoothed_height)))
                    self._cursor = search_start + offset
            else:
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
