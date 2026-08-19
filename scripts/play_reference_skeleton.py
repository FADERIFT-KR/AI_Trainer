#!/usr/bin/env python3
"""AI Hub 정상데이터 CSV/JSON 기반 정상 스쿼트 자세를 스켈레톤으로 실시간 재생한다.

웹캠/MediaPipe/lifting/DTW는 전혀 쓰지 않는다 (그 파이프라인은 지금 비활성화 상태).
기본값으로는 이미 구축된 Reference DB(output/reference_db/, AI Hub 3d_points.csv +
annotation.json으로 만든 Ground Truth Reference)의 "정상" medoid를 그대로 반복
재생해서 "정확한 스쿼트 자세가 어떤 모양인지" 눈으로 보여준다.

좌표는 정규화(Hip-centered + Scale + Orientation Alignment) 완료된 body-centered
좌표계의 lateral(좌우)-vertical(상하) 평면에 투영한다 — 특정 카메라와 무관하게
항상 "정면에서 본 사람" 모양이 되도록.

사용 예:
    # Reference DB의 정상 medoid #0 반복 재생 (기본값)
    python scripts/play_reference_skeleton.py

    # 정상 medoid #2로
    python scripts/play_reference_skeleton.py --medoid-rank 2

    # 다른 오류유형도 참고용으로 볼 수 있음
    python scripts/play_reference_skeleton.py --class 발뒤꿈치오류 --medoid-rank 0

    # Reference DB에 없는 임의의 actor/rep을 AI Hub zip에서 직접 재생
    python scripts/play_reference_skeleton.py \\
        --zip "<TL.zip 경로>" --actor CA01 --rep 2 --error-type 정상
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS  # noqa: E402
from ai_trainer.render import draw_skeleton_panel, fit_transform  # noqa: E402
from ai_trainer.reference_pipeline import build_ground_truth_reference  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "output" / "reference_db"
PANEL_W, PANEL_H = 640, 640
WINDOW_NAME = "정상 스쿼트 자세 Reference (q: quit)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class", dest="class_label", default="정상", choices=["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"])
    p.add_argument("--medoid-rank", type=int, default=0, help="Reference DB medoid 순번 (0~3)")
    p.add_argument("--tier", default="ground_truth", choices=["ground_truth", "operational"])
    p.add_argument("--fps", type=float, default=12.0)

    p.add_argument("--zip", default=None, help="지정하면 Reference DB 대신 이 zip에서 직접 시퀀스를 읽어 재생")
    p.add_argument("--actor", default=None)
    p.add_argument("--rep", default=None)
    p.add_argument("--error-type", default="정상", choices=["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"])
    p.add_argument("--level", default=None, choices=["초급", "중급", "고급"])
    return p.parse_args()


def load_from_reference_db(class_label: str, medoid_rank: int, tier: str) -> tuple[np.ndarray, dict, str]:
    manifest = json.loads((DB_DIR / "manifest.json").read_text(encoding="utf-8"))["entries"]
    arrays = np.load(DB_DIR / "sequences.npz")

    for e in manifest:
        rank = int(e["medoid_id"].split("_")[1])
        if e["class_label"] == class_label and rank == medoid_rank and e["tier"] == tier:
            coords = arrays[e["array_key"]]
            title = f"{class_label} #{medoid_rank} ({tier}) actor={e['actor_id']} level={e['difficulty_level']}"
            return coords, e["phase_boundaries"], title

    available = sorted({(e["class_label"], e["medoid_id"].split("_")[1]) for e in manifest if e["tier"] == tier})
    raise SystemExit(
        f"[오류] Reference DB에 {class_label} medoid #{medoid_rank} ({tier})가 없습니다.\n"
        f"사용 가능한 조합: {available}"
    )


def load_from_zip(args: argparse.Namespace) -> tuple[np.ndarray, dict, str]:
    with AiHubZip(args.zip) as z:
        matches = z.find_sequences(error_type=args.error_type, level=args.level, actor=args.actor, rep=args.rep)
        if not matches:
            raise SystemExit("[오류] 조건에 맞는 시퀀스가 없습니다.")
        seq = matches[0]
        ref = build_ground_truth_reference(z, seq, origin="zip")
        if ref is None:
            raise SystemExit(f"[오류] {seq} 시퀀스를 처리할 수 없습니다 (구간이 너무 짧음 등).")

    from ai_trainer.phase_features import extract_phase_features
    from ai_trainer.phase_segmentation import segment_phases

    bounds = segment_phases(extract_phase_features(ref.coords)).as_dict()
    title = f"{seq.error_type} {seq.actor}/rep{seq.rep} ({seq.level})"
    return ref.coords, bounds, title


def phase_at(bounds: dict, t: int) -> str:
    for phase, (s, e) in bounds.items():
        if s <= t < e:
            return phase
    return "-"


def main() -> None:
    args = parse_args()

    if args.zip:
        coords, bounds, title = load_from_zip(args)
    else:
        coords, bounds, title = load_from_reference_db(args.class_label, args.medoid_rank, args.tier)

    print(f"재생: {title}  ({coords.shape[0]}프레임, {args.fps}fps 반복재생)")
    print("웹캠/MediaPipe/DTW 파이프라인은 사용하지 않음 — 순수 AI Hub 데이터 재생")
    print("'q' 키로 종료")

    proj = coords[:, :, [0, 1]]  # lateral(좌우) - vertical(상하) 평면, body-centered (카메라 무관 정면 시점)
    tf = fit_transform(proj, PANEL_W, PANEL_H, flip_y=True)
    delay_ms = max(1, int(1000 / args.fps))

    t = 0
    n = coords.shape[0]
    while True:
        canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        footer = f"frame {t}/{n - 1}   phase={phase_at(bounds, t)}"
        draw_skeleton_panel(canvas, (0, 0), PANEL_W, PANEL_H, tf(proj[t]), title, footer, COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

        t = (t + 1) % n

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
