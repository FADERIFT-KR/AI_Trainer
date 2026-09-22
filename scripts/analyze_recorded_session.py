"""Inspect a saved squat session without needing the original webcam or models.

Usage: python scripts/analyze_recorded_session.py PATH_TO_squat_session_...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trainer.core.common_skeleton import COMMON_JOINT_NAMES


_COMMON_INDEX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}


def _world_joint(row: dict, index: int) -> np.ndarray | None:
    landmarks = row.get("world_landmarks")
    if not isinstance(landmarks, list) or len(landmarks) <= index:
        return None
    point = np.asarray(landmarks[index][:3], dtype=float)
    return point if point.shape == (3,) and np.isfinite(point).all() else None


def spike_candidates(rows: list[dict], joint_index: int = 25,
                     threshold_m: float = 0.12) -> list[dict]:
    """Flag isolated one-frame knee jumps; this is a review cue, not a diagnosis."""
    flagged = []
    for index in range(1, len(rows) - 1):
        before = _world_joint(rows[index - 1], joint_index)
        current = _world_joint(rows[index], joint_index)
        after = _world_joint(rows[index + 1], joint_index)
        if before is None or current is None or after is None:
            continue
        jump = float(np.linalg.norm(current - (before + after) / 2.0))
        neighbor_change = float(np.linalg.norm(after - before))
        if jump >= threshold_m and neighbor_change < jump * 0.65:
            flagged.append({"video_frame": rows[index]["video_frame"],
                            "elapsed_ms": rows[index]["elapsed_ms"],
                            "deviation_m": round(jump, 3)})
    return flagged


def occluded_display_jumps(rows: list[dict], joint_name: str, mp_index: int,
                           threshold: float = 0.30) -> list[dict]:
    """Find large displayed 3D jumps when the source landmark is occluded."""
    joint_index = _COMMON_INDEX[joint_name]
    flagged = []
    for index in range(1, len(rows)):
        before = rows[index - 1].get("display_aligned_3d")
        current = rows[index].get("display_aligned_3d")
        landmarks = rows[index].get("world_landmarks")
        if before is None or current is None or landmarks is None:
            continue
        visibility = float(landmarks[mp_index][3])
        first = np.asarray(before[joint_index], dtype=float)
        second = np.asarray(current[joint_index], dtype=float)
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            continue
        jump = float(np.linalg.norm(second - first))
        if visibility < 0.5 and jump >= threshold:
            flagged.append({"video_frame": rows[index]["video_frame"],
                            "elapsed_ms": rows[index]["elapsed_ms"],
                            "jump_normalized": round(jump, 3),
                            "visibility": round(visibility, 3)})
    return flagged


def analyze(directory: Path) -> dict:
    session_path = directory / "session.json"
    if not session_path.is_file():
        raise FileNotFoundError(f"session.json이 없습니다: {directory}")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    report = {"final_result": session.get("summary", {}).get("final_result"),
              "missing_views": session.get("missing_views", []), "views": {}}
    for view in session.get("views", []):
        name = view["view"]
        timeline = directory / view["timeline_file"]
        video = directory / view["video_file"]
        if not timeline.is_file() or not video.is_file():
            raise FileNotFoundError(f"{name} 영상 또는 프레임 기록이 없습니다")
        rows = [json.loads(line) for line in timeline.read_text(encoding="utf-8").splitlines()]
        if len(rows) != view["frame_count"] or any(row["video_frame"] != index
                                                    for index, row in enumerate(rows)):
            raise ValueError(f"{name} 영상 프레임과 분석 기록의 번호가 일치하지 않습니다")
        has_display_coordinates = any(row.get("display_aligned_3d") is not None for row in rows)
        report["views"][name] = {
            "video": str(video),
            "frames": len(rows),
            "duration_seconds": round(rows[-1]["elapsed_ms"] / 1000, 2) if rows else 0,
            "framing_coverage": round(sum(bool(row.get("framing_ok")) for row in rows) / len(rows), 3)
            if rows else 0.0,
            "display_tracking_available": has_display_coordinates,
            "repetitions": [
                {"video_frame": row["video_frame"], "elapsed_ms": row["elapsed_ms"],
                 **row["completed_rep"]}
                for row in rows if row.get("completed_rep") is not None
            ],
            "left_knee_spike_candidates": spike_candidates(rows, 25),
            "right_knee_spike_candidates": spike_candidates(rows, 26),
            "far_wrist_display_jumps": (
                occluded_display_jumps(rows, "RWrist", 16) if name == "left"
                else occluded_display_jumps(rows, "LWrist", 15) if name == "right"
                else []
            ) if has_display_coordinates else None,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="저장된 squat_session_* 폴더")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(analyze(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
