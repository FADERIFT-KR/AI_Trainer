"""2D→3D Temporal Lifting 학습 데이터셋 생성.

camera1(정면 후보) 2D CSV + 3D GT CSV를 프레임 인덱스로 정렬해
``(T, 18, 2) -> (18, 3)`` (center frame) 윈도우 샘플을 만든다.

- actor 단위 split(``configs/actor_split.json``)을 **시퀀스 단위**로 먼저 적용한 뒤
  윈도우를 생성한다 — 프레임 단위 랜덤 분할은 하지 않는다.
- 학습 샘플(center frame)은 annotation.json의 ``start_frame~end_frame``
  (실제 에어스쿼트 구간) 안에서만 뽑는다. 시간창(window) 문맥은 그 밖의
  실제 캡처 프레임(대기 구간)도 사용하되, 클립 경계를 벗어나면 edge-replication.
- raw(정규화 전)와 norm(정규화 후) 좌표를 모두 보존해 반환한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actor_split import load_all_air_squat_sequences
from .aihub_zip import AiHubZip, SequenceKey
from .common_skeleton import to_common_skeleton
from .normalization import hip_center_3d, normalize_2d_sequence

CAMERA = 1  # 정면 후보 (claude.md 7장: 다중 조건 검증으로 강하게 추정되는 카메라, GT 아님)
WINDOW_T = 9
FRAME_STRIDE = 2  # 인접 프레임 과다 중복을 줄이기 위한 center-frame 추출 간격


@dataclass
class WindowSample:
    x_raw: np.ndarray  # (T,18,2) 원본 픽셀 좌표
    x_norm: np.ndarray  # (T,18,2) 정규화된 좌표 (Hip root-center + 시퀀스 고정 스케일)
    y_raw: np.ndarray  # (18,3) 원본 3D 좌표 (center frame, 카메라 로컬)
    y_norm: np.ndarray  # (18,3) Hip-centered 3D (center frame)
    origin: str
    actor: str
    level: str
    error_type: str
    rep: str
    center_frame: int


def load_actor_split(split_path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    return payload["actor_to_split"]


def build_windows_for_sequence(
    z: AiHubZip, seq: SequenceKey, origin: str
) -> list[WindowSample]:
    try:
        frames_3d, coords_3d_26 = z.read_3d(seq)
        frames_2d, coords_2d_26 = z.read_2d(seq, CAMERA)
        ann = z.read_annotation(seq, CAMERA)
    except FileNotFoundError:
        return []

    n = min(len(frames_3d), len(frames_2d))
    if n < WINDOW_T:
        return []

    coords_3d = to_common_skeleton(coords_3d_26[:n])  # (n,18,3)
    coords_2d = to_common_skeleton(coords_2d_26[:n])  # (n,18,2)

    start_f, end_f = 0, n - 1
    if ann.get("annotations"):
        a0 = ann["annotations"][0]
        start_f = max(0, min(a0["start_frame"], n - 1))
        end_f = max(start_f, min(a0["end_frame"], n - 1))

    norm_2d_full = normalize_2d_sequence(coords_2d)  # 시퀀스(전체 클립) 고정 스케일 기준

    half = WINDOW_T // 2
    samples: list[WindowSample] = []
    for c in range(start_f, end_f + 1, FRAME_STRIDE):
        idxs = np.clip(np.arange(c - half, c + half + 1), 0, n - 1)
        y_raw = coords_3d[c]
        y_norm, _ = hip_center_3d(y_raw)

        samples.append(
            WindowSample(
                x_raw=coords_2d[idxs].astype(np.float32),
                x_norm=norm_2d_full[idxs].astype(np.float32),
                y_raw=y_raw.astype(np.float32),
                y_norm=y_norm.astype(np.float32),
                origin=origin,
                actor=seq.actor,
                level=seq.level,
                error_type=seq.error_type,
                rep=seq.rep,
                center_frame=int(c),
            )
        )
    return samples


def build_split_datasets(
    tl_zip: str | Path, vl_zip: str | Path, split_path: str | Path
) -> tuple[list[WindowSample], list[WindowSample]]:
    """actor_split.json 기준으로 train/val 시퀀스를 나누고, 각 시퀀스 전체를 윈도우로 변환한다."""
    actor_to_split = load_actor_split(split_path)
    sequences = load_all_air_squat_sequences(tl_zip, vl_zip)

    zips = {"TL": AiHubZip(tl_zip), "VL": AiHubZip(vl_zip)}
    train_samples: list[WindowSample] = []
    val_samples: list[WindowSample] = []
    try:
        for os_ in sequences:
            side = actor_to_split.get(os_.seq.actor)
            if side is None:
                continue  # split에 없는 actor(있어서는 안 되지만 방어적으로 스킵)
            samples = build_windows_for_sequence(zips[os_.origin], os_.seq, os_.origin)
            (train_samples if side == "train" else val_samples).extend(samples)
    finally:
        for z in zips.values():
            z.close()

    return train_samples, val_samples


def samples_to_arrays(samples: list[WindowSample]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("빈 샘플 리스트입니다.")
    return {
        "x_raw": np.stack([s.x_raw for s in samples]),
        "x_norm": np.stack([s.x_norm for s in samples]),
        "y_raw": np.stack([s.y_raw for s in samples]),
        "y_norm": np.stack([s.y_norm for s in samples]),
        "origin": np.array([s.origin for s in samples]),
        "actor": np.array([s.actor for s in samples]),
        "level": np.array([s.level for s in samples]),
        "error_type": np.array([s.error_type for s in samples]),
        "rep": np.array([s.rep for s in samples]),
        "center_frame": np.array([s.center_frame for s in samples]),
    }
