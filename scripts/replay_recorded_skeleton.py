"""Render saved camera frames beside recorded and recomputed 3D skeletons.

Example:
    python scripts/replay_recorded_skeleton.py SESSION_DIR --view right --output output/right_corrected.mp4

This reuses recorded MediaPipe landmarks, so it tests the tracking/framing
changes without a live camera or rerunning the pose model. The left skeleton is
recomputed from the saved landmarks through the current bridge, so it shows what
today's code produces for a session recorded earlier. The original saved
video and JSONL remain untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trainer.squat.camera_views import VIEWS
from ai_trainer.core.s3_mapping.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS
from ai_trainer.squat.game_ui.framing_check import check_framing
from ai_trainer.core.s3_mapping.pose_bridge import CommonSkeleton3DBridge
from ai_trainer.core.ui.panel_render import draw_skeleton_panel, fit_transform


PANEL_SIZE = 420
CAMERA_SIZE = (640, PANEL_SIZE)
FRAMING_DEBOUNCE_FRAMES = 5  # mirrors SquatPipelineWorker


def _project(coords: np.ndarray, view: str) -> np.ndarray:
    if view == "front":
        return coords[..., [0, 1]]
    projected = coords[..., [2, 1]].copy()
    if view == "right":
        projected[..., 0] *= -1
    return projected


def _calibration_matrix(rows: list[dict]) -> np.ndarray:
    """Recover the original session's fixed 3D alignment from JSONL pairs."""
    pairs = [(np.asarray(row["common_3d"], dtype=float),
              np.asarray(row["aligned_3d"], dtype=float))
             for row in rows if row.get("common_3d") is not None
             and row.get("aligned_3d") is not None][:8]
    if not pairs:
        raise ValueError("보정된 3D 기록이 없어 동일한 좌표계로 재생할 수 없습니다")
    source = np.concatenate([pair[0] for pair in pairs])
    target = np.concatenate([pair[1] for pair in pairs])
    matrix, _, _, _ = np.linalg.lstsq(source, target, rcond=None)
    if not np.isfinite(matrix).all():
        raise ValueError("저장된 3D 좌표로 화면 변환을 복원할 수 없습니다")
    return matrix


def _panel(points: np.ndarray | None, transform, title: str) -> np.ndarray:
    canvas = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    if points is None:
        cv2.putText(canvas, f"{title}: no skeleton", (20, PANEL_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
    else:
        draw_skeleton_panel(canvas, (0, 0), PANEL_SIZE, PANEL_SIZE,
                            transform(points), title, None,
                            COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR)
    return canvas


def replay(directory: Path, view: str, output: Path) -> dict:
    if view not in VIEWS:
        raise ValueError(f"지원하지 않는 촬영 시점: {view}")
    metadata = json.loads((directory / f"{view}.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / metadata["timeline_file"])
            .read_text(encoding="utf-8").splitlines()]
    if len(rows) != metadata["frame_count"]:
        raise ValueError("영상과 프레임 기록의 길이가 다릅니다")
    width, height = metadata["frame_size"]
    alignment = _calibration_matrix(rows)
    bridge = CommonSkeleton3DBridge()
    corrected = []
    for row in rows:
        if row.get("world_landmarks") is None or row.get("image_landmarks") is None:
            corrected.append(None)
            continue
        common, _, _ = bridge.update(np.asarray(row["world_landmarks"], dtype=float))
        corrected.append(_project(common @ alignment, view))
    original = [_project(np.asarray(row["aligned_3d"], dtype=float), view)
                if row.get("aligned_3d") is not None else None for row in rows]
    all_points = [points for points in corrected + original if points is not None]
    if not all_points:
        raise ValueError("재생할 3D 스켈레톤 좌표가 없습니다")
    transform = fit_transform(np.stack(all_points), PANEL_SIZE, PANEL_SIZE, margin=32)

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"출력 영상이 이미 있습니다: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(directory / metadata["video_file"]))
    if not capture.isOpened():
        raise OSError("저장된 카메라 영상을 열 수 없습니다")
    canvas_size = (CAMERA_SIZE[0] + PANEL_SIZE * 2, PANEL_SIZE)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(metadata["video_fps"]), canvas_size)
    if not writer.isOpened():
        capture.release()
        raise OSError("MP4 출력 코덱을 열 수 없습니다")
    new_gate_count = 0
    raw_gate_count = 0
    effective_gate = bool(rows[0]["framing_ok"])
    gate_streak = 0
    try:
        for index, row in enumerate(rows):
            success, frame = capture.read()
            if not success:
                raise ValueError(f"영상 {index}번째 프레임을 읽지 못했습니다")
            raw_gate = (row.get("image_landmarks") is not None and
                        check_framing(np.asarray(row["image_landmarks"]), width, height,
                                      relax_distance=True, view=view).ok)
            raw_gate_count += bool(raw_gate)
            if raw_gate == effective_gate:
                gate_streak = 0
            else:
                gate_streak += 1
                if gate_streak >= FRAMING_DEBOUNCE_FRAMES:
                    effective_gate = bool(raw_gate)
                    gate_streak = 0
            new_gate_count += effective_gate
            camera = cv2.resize(frame, CAMERA_SIZE)
            cv2.putText(camera, f"{view} frame {index}  {row['elapsed_ms'] / 1000:.2f}s",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(camera, f"old gate: {row['framing_ok']}  new gate: {effective_gate}",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (70, 240, 70) if effective_gate else (70, 70, 240), 2)
            frame_out = np.hstack((camera,
                                   _panel(original[index], transform, "Saved analysis"),
                                   _panel(corrected[index], transform, "Recomputed (current code)")))
            writer.write(frame_out)
    finally:
        capture.release()
        writer.release()
    return {"output": str(output), "frames": len(rows),
            "saved_gate_frames": sum(bool(row["framing_ok"]) for row in rows),
            "corrected_raw_gate_frames": raw_gate_count,
            "corrected_gate_frames": new_gate_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="squat_session_* 폴더")
    parser.add_argument("--view", choices=VIEWS, default="right")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.directory, args.view, args.output), ensure_ascii=True))


if __name__ == "__main__":
    main()
