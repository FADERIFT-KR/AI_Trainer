#!/usr/bin/env python3
"""AI Hub 에어스쿼트 라벨링 데이터(zip)만으로 스켈레톤 확인 영상을 만든다.

원본 영상/이미지는 전혀 사용하지 않는다. TL.zip / VL.zip 안의
``3d_points.csv``(3D), ``camera{N}/local_keypoints/*.csv``(2D),
``camera{N}/video/annotation.json``(스쿼트 동작 구간)만 읽어서
2D 정면 카메라 스켈레톤 + 3D 좌표를 3면(X-Y/X-Z/Y-Z)으로 투영한
합성 영상(mp4)을 만든다. 3D 축이 아직 무엇을 의미하는지(상하/좌우/전후)
확정되지 않았으므로, 3면을 모두 보여줘서 눈으로 축 방향을 확인할 수 있게 한다.

사용 예:

    # 조건에 맞는 시퀀스 목록만 확인
    python scripts/visualize_skeleton.py --zip .../TL.zip \\
        --error-type 정상 --actor CA01 --list

    # 실제 영상 생성 (기본: 정면 후보 camera1 + 3D 3면도 합성)
    python scripts/visualize_skeleton.py --zip .../TL.zip \\
        --error-type 정상 --actor CA01 --rep 2 \\
        --out output/CA01_정상_rep2.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.render import draw_skeleton_panel, fit_transform  # noqa: E402

PANEL_W, PANEL_H = 480, 400
HEADER_H = 28
GRID_W, GRID_H = PANEL_W * 2, PANEL_H * 2 + HEADER_H


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--zip", required=True, help="TL.zip 또는 VL.zip 경로")
    p.add_argument(
        "--error-type",
        default="정상",
        choices=["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"],
    )
    p.add_argument("--level", default=None, choices=["초급", "중급", "고급"])
    p.add_argument("--actor", default=None, help="예: CA01")
    p.add_argument("--rep", default=None, help="반복 인덱스, 예: 1")
    p.add_argument(
        "--camera", type=int, default=1, help="2D 시각화용 카메라 번호 (기본값 1 = 정면 후보)"
    )
    p.add_argument(
        "--out",
        default=None,
        help="출력 mp4 경로 (기본: output/<actor>_<오류유형>_rep<rep>_cam<N>.mp4)",
    )
    p.add_argument(
        "--fps", type=float, default=30.0,
        help="출력 영상 fps (AI Hub 원본 캡처가 30fps로 확인됨 — annotation.json의 "
        "start_time/end_time 대비 start_frame/end_frame 역산 검증 완료; 기본 30)",
    )
    p.add_argument(
        "--list", action="store_true", help="조건에 맞는 시퀀스 목록만 출력하고 종료"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with AiHubZip(args.zip) as z:
        matches = z.find_sequences(
            error_type=args.error_type, level=args.level, actor=args.actor, rep=args.rep
        )
        if not matches:
            print("조건에 맞는 시퀀스가 없습니다.", file=sys.stderr)
            sys.exit(1)

        if args.list:
            for m in matches:
                print(f"{m}  cameras={z.list_cameras(m)}")
            return

        seq = matches[0]
        if len(matches) > 1:
            print(f"[안내] 조건에 맞는 시퀀스 {len(matches)}개 중 첫 번째를 사용합니다: {seq}")
        else:
            print(f"시퀀스: {seq}")

        cams = z.list_cameras(seq)
        if args.camera not in cams:
            print(
                f"[오류] camera{args.camera}가 이 시퀀스에 없습니다. 존재하는 카메라: {cams}",
                file=sys.stderr,
            )
            sys.exit(1)

        frames_3d, coords_3d = z.read_3d(seq)  # (T, 26, 3)
        frames_2d, coords_2d = z.read_2d(seq, args.camera)  # (T, 26, 2)
        ann = z.read_annotation(seq, args.camera)

    start_f = end_f = None
    if ann.get("annotations"):
        start_f = ann["annotations"][0]["start_frame"]
        end_f = ann["annotations"][0]["end_frame"]
    actor_info = ann.get("actor", {})

    n_frames = min(len(frames_3d), len(frames_2d))
    if len(frames_3d) != len(frames_2d):
        print(
            f"[경고] 3D({len(frames_3d)})와 2D({len(frames_2d)}) 프레임 수가 달라 "
            f"{n_frames}프레임까지만 사용합니다."
        )

    # 시퀀스 전체 범위 기준으로 스케일을 고정해 프레임 간 화면이 흔들리지 않게 한다.
    tf_2d = fit_transform(coords_2d[:n_frames], PANEL_W, PANEL_H, flip_y=True)
    tf_xy = fit_transform(coords_3d[:n_frames][:, :, [0, 1]], PANEL_W, PANEL_H, flip_y=True)
    tf_xz = fit_transform(coords_3d[:n_frames][:, :, [0, 2]], PANEL_W, PANEL_H, flip_y=False)
    tf_yz = fit_transform(coords_3d[:n_frames][:, :, [1, 2]], PANEL_W, PANEL_H, flip_y=True)

    out_path = (
        Path(args.out)
        if args.out
        else Path("output")
        / f"{seq.actor}_{seq.error_type}_rep{seq.rep}_cam{args.camera}.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (GRID_W, GRID_H))
    if not writer.isOpened():
        print(f"[오류] VideoWriter를 열 수 없습니다: {out_path}", file=sys.stderr)
        sys.exit(1)

    header = (
        f"{seq.actor}({actor_info.get('actor_height', '?')}cm)  "
        f"{seq.error_type}/{seq.level}  rep{seq.rep}  camera{args.camera}"
    )

    for t in range(n_frames):
        canvas = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
        in_phase = start_f is not None and start_f <= t <= end_f
        footer = f"frame {t}/{n_frames - 1}  {'[동작구간]' if in_phase else '[대기구간]'}"

        draw_skeleton_panel(
            canvas, (0, 0), PANEL_W, PANEL_H,
            tf_2d(coords_2d[t]), f"2D camera{args.camera} (정면 후보)", footer,
        )
        draw_skeleton_panel(
            canvas, (PANEL_W, 0), PANEL_W, PANEL_H,
            tf_xy(coords_3d[t][:, [0, 1]]), "3D  X-Y", footer,
        )
        draw_skeleton_panel(
            canvas, (0, PANEL_H), PANEL_W, PANEL_H,
            tf_xz(coords_3d[t][:, [0, 2]]), "3D  X-Z (탑뷰 추정)", footer,
        )
        draw_skeleton_panel(
            canvas, (PANEL_W, PANEL_H), PANEL_W, PANEL_H,
            tf_yz(coords_3d[t][:, [1, 2]]), "3D  Y-Z (측면 추정)", footer,
        )

        cv2.putText(
            canvas, header, (8, GRID_H - HEADER_H + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )
        writer.write(canvas)

    writer.release()
    print(f"완료: {out_path}  ({n_frames}프레임, {args.fps}fps)")


if __name__ == "__main__":
    main()
