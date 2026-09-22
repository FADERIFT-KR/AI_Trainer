"""Replay recorded MediaPipe landmarks through the current game session."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_trainer.game_ui.pose_bridge import CommonSkeleton3DBridge
from ai_trainer.dl_classifier import DLSquatClassifier
from ai_trainer.online_dtw import OnlineSquatSession
from ai_trainer.reference_db_io import load_reference_db
from ai_trainer.scoring import PASS_SCORE_THRESHOLD


def replay(records, *, timed=True):
    config = json.loads((ROOT / "configs/dtw_feature_weights.json").read_text(encoding="utf-8"))
    session = OnlineSquatSession(torch.nn.Identity(), torch.device("cpu"), load_reference_db(ROOT / "output/reference_db")["ground_truth"], config)
    checkpoint, norm = ROOT / "output/dl_classifier/model.pt", ROOT / "output/dl_classifier/norm_stats.npz"
    if checkpoint.exists() and norm.exists():
        session.dl_classifier = DLSquatClassifier.load(checkpoint, norm)
    calibration = ROOT / "output/dtw_eval/offline_eval_report.json"
    if calibration.exists():
        session.score_calib = json.loads(calibration.read_text(encoding="utf-8"))["score_calibration_ground_truth"]
    session._partial_online_distance = lambda t: None
    session._joint_feedback_frame = lambda t: None
    bridge = CommonSkeleton3DBridge(min_visibility=0.4)
    for r in records:
        timestamp = r["timestamp"] if timed else None
        common, _, _ = bridge.update(np.array(r["world_landmarks"]), timestamp=timestamp)
        session.push_frame_3d(common, timestamp=timestamp)
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--legacy-timing", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines()]
    session = replay(records, timed=not args.legacy_timing)
    report = {"reps": [asdict(r) for r in session.completed_reps], "processed_samples": len(session.aligned_seq),
              "display_results": ["정상" if r.score_vs_normal is not None and r.score_vs_normal >= PASS_SCORE_THRESHOLD else r.predicted_class for r in session.completed_reps]}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
