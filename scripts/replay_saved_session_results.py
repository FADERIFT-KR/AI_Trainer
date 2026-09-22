"""Derive the current split result screen from a saved squat session.

The recorder stores the webcam video together with the common 2D skeleton and
the completed-repetition event for every view.  This tool replays those stored
frames (it does not run MediaPipe again), retains the original DTW/ML result,
and evaluates the current paper bottom-pose conditions independently.

Example:
    python scripts/replay_saved_session_results.py \
        C:\\Users\\user\\Desktop\\TP\\saved\\squat_session_20260917_213840
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trainer.squat.camera_views import VIEW_FRONT, VIEW_LABEL_KO, VIEWS
from ai_trainer.squat.session_decision import decide_session, format_session_decision
from ai_trainer.squat.view_conditions import assess_paper_squat_conditions


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _paper_result(view: str, violations: tuple) -> tuple[str, list[str]]:
    messages = [violation.message for violation in violations]
    if view == VIEW_FRONT:
        return "미평가 — 정면에서는 전후 깊이·발끝 대비 무릎 위치를 판별할 수 없음", messages
    if messages:
        return "위반 — " + " / ".join(messages), messages
    return "통과 — 고관절 각도·허벅지 수평·무릎-발끝 정렬", messages


def replay_session(directory: Path) -> dict:
    """Return records formatted like the application's split result screen."""
    session_path = directory / "session.json"
    if not session_path.is_file():
        raise FileNotFoundError(f"session.json이 없습니다: {directory}")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    summary = session.get("summary", {})
    previous_reps = summary.get("repetitions", [])
    previous_by_view_rep = {
        (item.get("view_mode"), int(item.get("view_rep_index", -1))): item
        for item in previous_reps
    }

    replay_reps: list[dict] = []
    missing: list[str] = []
    for view_info in session.get("views", []):
        view = view_info.get("view")
        if view not in VIEWS:
            continue
        timeline_path = directory / str(view_info.get("timeline_file", f"{view}.jsonl"))
        video_path = directory / str(view_info.get("video_file", f"{view}.mp4"))
        if not timeline_path.is_file() or not video_path.is_file():
            missing.append(view)
            continue
        rows = _read_jsonl(timeline_path)
        events = [row["completed_rep"] for row in rows if row.get("completed_rep")]
        for event in events:
            rep_index = int(event.get("rep_index", 0))
            saved = previous_by_view_rep.get((view, rep_index), {})
            frame_range = event.get("frame_range", [0, -1])
            if not isinstance(frame_range, list) or len(frame_range) != 2:
                raise ValueError(f"{view} REP {rep_index + 1}의 프레임 범위가 잘못되었습니다")
            start, end = map(int, frame_range)
            if start < 0 or end < start or end >= len(rows):
                raise ValueError(
                    f"{view} REP {rep_index + 1}의 범위 {frame_range}가 저장 프레임 {len(rows)}개를 벗어납니다"
                )
            valid_frames = []
            dropped_2d_frames = 0
            for row in rows[start:end + 1]:
                try:
                    frame = np.asarray(row["common_2d"], dtype=float)
                except (KeyError, TypeError, ValueError):
                    frame = np.empty((0, 2))
                if frame.shape == (18, 2) and np.isfinite(frame).all():
                    valid_frames.append(frame)
                else:
                    # Some older recordings contain an empty common_2d field
                    # during a transient tracker failure.  It must not make a
                    # whole saved session impossible to review.
                    dropped_2d_frames += 1
            if len(valid_frames) < 2:
                raise ValueError(f"{view} REP {rep_index + 1}에 재생 가능한 2D 관절이 2프레임 미만입니다")
            coords = np.stack(valid_frames)

            # Old recordings did not preserve the phase boundaries.  The
            # evaluator then selects the deepest visible knee frame itself.
            violations = assess_paper_squat_conditions(coords, {}, view)
            paper_result, paper_messages = _paper_result(view, violations)
            assessment = deepcopy(saved.get("condition_assessment") or event.get("condition_assessment") or {})
            assessment["paper_posture_violations"] = [
                {
                    "condition": violation.condition,
                    "value": violation.value,
                    "lower": violation.lower,
                    "upper": violation.upper,
                    "message": violation.message,
                }
                for violation in violations
            ]
            sequence_class = str(saved.get("sequence_class") or saved.get("display_class")
                                 or event.get("predicted_class") or "판정 불확실")
            display_class = "논문 자세 기준 위반" if paper_messages and sequence_class == "정상" else sequence_class
            replay_reps.append({
                "index": len(replay_reps),
                "view_rep_index": rep_index,
                "view_mode": view,
                "frame_range": [start, end],
                "paper_input_frames": len(valid_frames),
                "paper_dropped_2d_frames": dropped_2d_frames,
                "video_frame": next(
                    row["video_frame"] for row in rows if row.get("completed_rep") is event
                ),
                "display_class": display_class,
                "sequence_class": sequence_class,
                "model_class": str(saved.get("model_class") or event.get("predicted_class") or "판정 불확실"),
                "score_text": str(saved.get("score_text") or _gate_text(event)),
                "fail_reason": str(saved.get("fail_reason") or ""),
                "paper_posture_result": paper_result,
                "paper_posture_messages": paper_messages,
                "condition_assessment": assessment,
            })

    decision = decide_session(replay_reps)
    return {
        "source_session": str(directory),
        "replay_method": "Stored camera video timeline and common_2d; MediaPipe was not rerun.",
        "caveat": (
            "Existing trajectory/ML results are retained from the recording. "
            "They are not a recalibration of the saved session's 3D body axis."
        ),
        "missing_views": missing,
        "repetitions": replay_reps,
        "final_result": format_session_result(decision, replay_reps),
    }


def _gate_text(event: dict) -> str:
    distance, threshold = event.get("gate_distance"), event.get("gate_threshold")
    if distance is not None and threshold is not None:
        return f"DTW {float(distance):.3f} / 기준 {float(threshold):.3f}"
    return "-"


def format_session_result(decision, repetitions: list[dict]) -> str:
    """Use the same two independent result sections as the GUI."""
    lines = [format_session_decision(decision), ""]
    for rep in repetitions:
        view_label = VIEW_LABEL_KO[rep["view_mode"]]
        lines.extend([
            f"{view_label} REP {rep['view_rep_index'] + 1}",
            f"  기존 궤적·ML 판정: {rep['sequence_class']} ({rep['score_text']})",
            f"  논문 최저점 자세 기준: {rep['paper_posture_result']}",
        ])
        if rep["fail_reason"]:
            lines.append(f"  기존 판정 근거: {rep['fail_reason']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="저장된 squat_session_* 폴더")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output") / "replay_saved",
        help="원본을 건드리지 않고 재생 결과를 저장할 폴더",
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = replay_session(args.directory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.directory.name + "_split_result"
    json_path = args.output_dir / f"{stem}.json"
    text_path = args.output_dir / f"{stem}.txt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(result["final_result"] + "\n", encoding="utf-8")
    print(result["final_result"])
    print(f"\n결과 저장: {json_path}\n텍스트 결과: {text_path}")


if __name__ == "__main__":
    main()
