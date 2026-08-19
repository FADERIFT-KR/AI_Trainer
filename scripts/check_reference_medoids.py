#!/usr/bin/env python3
"""선정된 Reference DB medoid들을 최저점(phase='최저점') 프레임에서 시각화해
실제로 합리적인 자세인지 확인한다.

정규화(Hip-center+Scale+Orientation) 좌표의 lateral(axis0)-vertical(axis1) 평면은
camera1 등 특정 카메라와 무관하게 항상 "몸 정면"처럼 보이므로 이 평면에 투영한다.
GT Reference(파랑 계열, 원래 뼈대 색)와 Operational Reference(주황 윤곽선)를 겹쳐 그려
lifting 모델의 systematic error가 얼마나 되는지도 함께 확인한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ai_trainer.common_skeleton import COMMON_BONE_INDEX_PAIRS  # noqa: E402
from ai_trainer.render import draw_skeleton_panel, fit_transform  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "output" / "reference_db"
OUT_PATH = DB_DIR / "medoid_check.png"

PANEL_W, PANEL_H = 260, 320
CLASSES = ["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"]


def main() -> None:
    manifest = json.loads((DB_DIR / "manifest.json").read_text(encoding="utf-8"))["entries"]
    arrays = np.load(DB_DIR / "sequences.npz")

    by_class_rank: dict[tuple[str, int], dict[str, dict]] = {}
    for e in manifest:
        rank = int(e["medoid_id"].split("_")[1])
        key = (e["class_label"], rank)
        by_class_rank.setdefault(key, {})[e["tier"]] = e

    k = max(r for _, r in by_class_rank.keys()) + 1
    canvas = np.zeros((PANEL_H * len(CLASSES), PANEL_W * k, 3), dtype=np.uint8)

    for row, cls in enumerate(CLASSES):
        for rank in range(k):
            entry = by_class_rank.get((cls, rank))
            if not entry:
                continue
            gt_e = entry.get("ground_truth")
            op_e = entry.get("operational")

            x0, y0 = rank * PANEL_W, row * PANEL_H
            if gt_e is None:
                continue

            gt_coords = arrays[gt_e["array_key"]]  # (T,18,3)
            bottom_s, bottom_e = gt_e["phase_boundaries"]["최저점"]
            bottom_t = (bottom_s + bottom_e) // 2
            gt_pts = gt_coords[bottom_t][:, [0, 1]]  # lateral-vertical

            tf = fit_transform(gt_coords[:, :, [0, 1]], PANEL_W, PANEL_H, flip_y=True)
            title = f"{cls[:4]} #{rank} {gt_e['actor_id']}"
            footer = f"n_cluster={gt_e['cluster_size']}"
            draw_skeleton_panel(canvas, (x0, y0), PANEL_W, PANEL_H, tf(gt_pts), title, footer, COMMON_BONE_INDEX_PAIRS)

            if op_e is not None:
                op_coords = arrays[op_e["array_key"]]
                ob_s, ob_e = op_e["phase_boundaries"]["최저점"]
                ob_t = (ob_s + ob_e) // 2
                op_pts = tf(op_coords[ob_t][:, [0, 1]])
                valid = np.isfinite(op_pts).all(axis=-1)
                for i, j in COMMON_BONE_INDEX_PAIRS:
                    if valid[i] and valid[j]:
                        p1 = tuple(op_pts[i].astype(int))
                        p2 = tuple(op_pts[j].astype(int))
                        cv2.line(canvas, (p1[0] + x0, p1[1] + y0), (p2[0] + x0, p2[1] + y0), (0, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "skeleton line = GT Reference, orange line = Operational Reference (same frame)", (10, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_PATH), canvas)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
