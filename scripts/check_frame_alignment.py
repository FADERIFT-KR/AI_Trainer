#!/usr/bin/env python3
"""학습 전 frame alignment 재검증: camera1 2D center frame과 3D GT(Y-Z projection)를
같은 샘플에서 나란히 그려 프레임이 실제로 맞는 자세를 가리키는지 눈으로 확인한다.

output/lifting_dataset/train.npz, val.npz에서 여러 actor/오류유형에 걸쳐 무작위로
샘플을 뽑아 PNG 그리드 한 장으로 저장한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ai_trainer.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS  # noqa: E402
from ai_trainer.render import draw_skeleton_panel, fit_transform  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "output" / "lifting_dataset"
OUT_PATH = ROOT / "output" / "frame_alignment_check.png"

PANEL_W, PANEL_H = 300, 300
N_SAMPLES = 12  # 4열 x 3행, 각 셀에 2D/3D 패널 2개
COLS = 4


def main() -> None:
    d = np.load(DATA_DIR / "train.npz", allow_pickle=True)
    n = d["x_raw"].shape[0]
    rng = np.random.default_rng(0)

    # actor가 최대한 겹치지 않도록 다양하게 표본 추출
    actors = d["actor"]
    unique_actor_idx = {}
    for i in rng.permutation(n):
        a = str(actors[i])
        if a not in unique_actor_idx:
            unique_actor_idx[a] = i
        if len(unique_actor_idx) >= N_SAMPLES:
            break
    idxs = list(unique_actor_idx.values())[:N_SAMPLES]

    rows = (len(idxs) + COLS - 1) // COLS
    cell_w = PANEL_W * 2
    cell_h = PANEL_H
    canvas = np.zeros((cell_h * rows, cell_w * COLS, 3), dtype=np.uint8)

    center_t = d["x_raw"].shape[1] // 2

    for k, i in enumerate(idxs):
        x_raw = d["x_raw"][i]  # (T,18,2)
        y_raw = d["y_raw"][i]  # (18,3)
        actor = str(d["actor"][i])
        err = str(d["error_type"][i])
        level = str(d["level"][i])
        frame = int(d["center_frame"][i])
        origin = str(d["origin"][i])

        pts_2d = x_raw[center_t]  # (18,2) camera1 pixel 좌표
        pts_3d_yz = y_raw[:, [1, 2]]  # (18,2) Y-Z projection (hip-centered 전, 원본 스케일)

        tf_2d = fit_transform(pts_2d[None], PANEL_W, PANEL_H, flip_y=True)
        tf_3d = fit_transform(pts_3d_yz[None], PANEL_W, PANEL_H, flip_y=True)

        col, row = k % COLS, k // COLS
        x0, y0 = col * cell_w, row * cell_h
        title = f"{origin}/{actor}/{err[:4]}/f{frame}"
        draw_skeleton_panel(
            canvas, (x0, y0), PANEL_W, PANEL_H, tf_2d(pts_2d),
            f"2D cam1 {title}", None, COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
        )
        draw_skeleton_panel(
            canvas, (x0 + PANEL_W, y0), PANEL_W, PANEL_H, tf_3d(pts_3d_yz),
            "3D Y-Z proj", None, COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), canvas)
    print(f"저장: {OUT_PATH}  ({len(idxs)}개 샘플, actor: {[str(d['actor'][i]) for i in idxs]})")


if __name__ == "__main__":
    main()
