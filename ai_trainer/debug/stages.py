"""스켈레톤 처리 공정 정의 + 워커→디버그 창 스냅샷 전달."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

# kind: 좌표 해석 방식
#   image   — (H,W,3) BGR 이미지 자체 (이미 스켈레톤이 그려진 카메라 프레임)
#   px2d    — (J,2) 픽셀 좌표, frame_size 기준
#   mp3d    — (33,3) MediaPipe world 좌표(미터, y 아래 방향)
#   m3d     — (18,3) Common 3D(미터, y 아래 방향, hip 근처 원점)
#   norm3d  — (18,3) 방향 정렬 + 다리길이 정규화 좌표(y 위 방향, hip 원점)
@dataclass(frozen=True)
class StageDef:
    id: str
    title: str
    description: str
    kind: str
    unit: str


STAGES: tuple[StageDef, ...] = (
    StageDef("mp_image_2d", "MediaPipe 2D", "카메라 프레임 위 33관절 원본 추정", "image", "px"),
    StageDef("mp_world_3d", "MediaPipe 3D 원본", "world_landmarks 33관절 (미터)", "mp3d", "m"),
    StageDef("common2d_raw", "Common 2D 매핑", "33→18관절 픽셀 좌표, 필터 전", "px2d", "px"),
    StageDef("common2d_filtered", "2D One-Euro 후", "화면 오버레이용 2D (common2d)", "px2d", "px"),
    StageDef("common3d_raw", "Common 3D 매핑", "33→18관절 3D, 필터 전", "m3d", "m"),
    StageDef("common3d_spike", "스파이크 가드 후", "Live3DSpikeGuard 통과", "m3d", "m"),
    StageDef("common3d_smooth", "One-Euro 후 · 판정 입력", "hip 원점 재중심화 (common3d)", "m3d", "m"),
    StageDef("analysis_aligned", "정렬·정규화 · 판정용", "R_body 회전 + 다리길이 스케일", "norm3d", "L"),
    StageDef("display_aligned_pre", "표시용 정렬", "stabilize_feet 브릿지 → 정렬", "norm3d", "L"),
    StageDef("final_display", "최종 화면 스켈레톤", "ImageGuided 보정 후 출력", "norm3d", "L"),
)

STAGE_BY_ID = {s.id: s for s in STAGES}


@dataclass
class StageSnapshot:
    stage_id: str
    data: np.ndarray | None
    note: str = ""
    frame_size: tuple[int, int] | None = None  # (w, h) — px2d 해석용


class StageTap(QObject):
    """워커 스레드가 프레임마다 공정별 스냅샷을 채우고 flush()로 UI에 넘긴다."""

    frame_ready = pyqtSignal(object)  # dict[str, StageSnapshot] + "_meta"

    def __init__(self) -> None:
        super().__init__()
        self._current: dict[str, object] = {}

    def put(
        self,
        stage_id: str,
        data: np.ndarray | None,
        note: str = "",
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        if stage_id not in STAGE_BY_ID:
            raise KeyError(stage_id)
        payload = None if data is None else np.array(data, copy=True)
        self._current[stage_id] = StageSnapshot(stage_id, payload, note, frame_size)

    def meta(self, **values: object) -> None:
        self._current.setdefault("_meta", {}).update(values)  # type: ignore[union-attr]

    def flush(self) -> None:
        frame = self._current
        self._current = {}
        self.frame_ready.emit(frame)


__all__ = ["STAGES", "STAGE_BY_ID", "StageDef", "StageSnapshot", "StageTap"]
