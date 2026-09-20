#!/usr/bin/env python3
"""Train side-view squat scoring without camera-far or arm-joint evidence.

The frontal artifacts remain unchanged. This script writes a separate pair of
side-view DTW thresholds and a graph checkpoint trained with the same fixed
left/right visibility masks used during live inference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from ai_trainer.actor_split import load_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.camera_views import VIEW_LEFT, VIEW_RIGHT  # noqa: E402
from ai_trainer.dataset_config import DATASET_PATH  # noqa: E402
from ai_trainer.mt_stgcn import MultiTaskSTGCN  # noqa: E402
from ai_trainer.reference_pipeline import build_ground_truth_reference  # noqa: E402
from ai_trainer.two_stage_squat import (  # noqa: E402
    ERROR_CLASSES, PART_TARGETS, NormalTemplateGate, choose_threshold, mask_occluded_joints,
)
from scripts.train_two_stage_squat import BATCH_SIZE, EPOCHS, SEED, metrics, time_warp  # noqa: E402

OUT = ROOT / "output" / "two_stage_squat"
SIDES = (VIEW_LEFT, VIEW_RIGHT)


def run() -> dict:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(SEED)
    base_report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    fit_actors = set(base_report["fit_actors"])
    calibration_actors = set(base_report["calibration_actors"])
    outer_actors = set(base_report["outer_validation_actors"])
    with np.load(OUT / "normal_template.npz") as saved:
        template = saved["track"]

    records = []
    with AiHubZip(DATASET_PATH) as dataset:
        for item in load_air_squat_sequences(DATASET_PATH):
            if item.seq.actor not in fit_actors | calibration_actors | outer_actors:
                continue
            ref = build_ground_truth_reference(dataset, item.seq, item.origin)
            if ref is not None and len(ref.coords) >= 10:
                records.append({"actor": item.seq.actor, "label": item.seq.error_type,
                                "track": ref.coords})
    cal = [record for record in records if record["actor"] in calibration_actors]
    fit_errors = [record for record in records
                  if record["actor"] in fit_actors and record["label"] in ERROR_CLASSES]
    cal_errors = [record for record in cal if record["label"] in ERROR_CLASSES]
    outer = [record for record in records if record["actor"] in outer_actors]
    if not cal or not fit_errors or not cal_errors or not outer:
        raise ValueError("Side-view training needs fit, calibration, and held-out records")

    gates = {}
    gate_metrics = {}
    for view in SIDES:
        temporary = NormalTemplateGate(template, 1.0, view)
        distances = np.asarray([temporary.assess(record["track"]).distance for record in cal])
        threshold, stats = choose_threshold(distances,
                                            np.asarray([record["label"] == "정상" for record in cal]))
        gates[view] = NormalTemplateGate(template, threshold, view)
        gate_metrics[view] = stats

    x_fit, y_fit, p_fit = [], [], []
    x_cal, y_cal = [], []
    for view in SIDES:
        gate = gates[view]
        for record in fit_errors:
            x_fit.append(mask_occluded_joints(gate.assess(record["track"]).aligned, view))
            y_fit.append(ERROR_CLASSES.index(record["label"]))
            p_fit.append(PART_TARGETS[record["label"]])
        for record in cal_errors:
            x_cal.append(mask_occluded_joints(gate.assess(record["track"]).aligned, view))
            y_cal.append(ERROR_CLASSES.index(record["label"]))
    x_fit = np.asarray(x_fit, dtype=np.float32)
    y_fit = np.asarray(y_fit, dtype=np.int64)
    p_fit = np.asarray(p_fit, dtype=np.float32)
    x_cal = np.asarray(x_cal, dtype=np.float32)
    y_cal = np.asarray(y_cal, dtype=np.int64)

    class_weights = len(y_fit) / (len(ERROR_CLASSES) * np.maximum(
        np.bincount(y_fit, minlength=len(ERROR_CLASSES)), 1))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    bce = nn.BCEWithLogitsLoss()
    model = MultiTaskSTGCN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best_accuracy, best_epoch, best_state = -1.0, 0, None
    for epoch in range(EPOCHS):
        model.train()
        batches = np.array_split(rng.permutation(len(x_fit)),
                                 max(1, int(np.ceil(len(x_fit) / BATCH_SIZE))))
        for indices in batches:
            augmented = np.stack([time_warp(x_fit[index], rng) for index in indices])
            logits, parts = model(torch.from_numpy(augmented.transpose(0, 3, 1, 2)))
            loss = ce(logits, torch.from_numpy(y_fit[indices])) + 0.25 * bce(
                parts, torch.from_numpy(p_fit[indices]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(x_cal.transpose(0, 3, 1, 2)))
            accuracy = float(np.mean(logits.argmax(1).numpy() == y_cal))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            best_state = {name: value.cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    validation = {}
    with torch.no_grad():
        for view in SIDES:
            predicted = []
            for record in outer:
                result = gates[view].assess(record["track"])
                if result.passed:
                    predicted.append("정상")
                else:
                    masked = mask_occluded_joints(result.aligned, view)
                    logits, _ = model(torch.from_numpy(masked.transpose(2, 0, 1)[None]))
                    predicted.append(ERROR_CLASSES[int(logits.argmax(1))])
            labels = np.asarray([record["label"] for record in outer])
            predicted = np.asarray(predicted)
            validation[view] = metrics(labels, predicted) | {
                "confusion": {actual: {guess: int(np.sum((labels == actual) & (predicted == guess)))
                                       for guess in ("정상", *ERROR_CLASSES)}
                              for actual in ("정상", *ERROR_CLASSES)},
            }

    report = {
        "source": str(DATASET_PATH),
        "fit_actors": sorted(fit_actors),
        "calibration_actors": sorted(calibration_actors),
        "outer_validation_actors": sorted(outer_actors),
        "thresholds": {view: gates[view].threshold for view in SIDES},
        "calibration_metrics": gate_metrics,
        "graph_best_epoch": best_epoch,
        "graph_calibration_error_accuracy": best_accuracy,
        "outer_validation": validation,
        "input_policy": "camera-near shoulder, hip, knee, ankle, heel and toe only; far and arm joints zeroed",
        "validation_usage_note": "Exploratory actor holdout inherited from the frontal study; not a sealed test.",
        "caveat": "3D CSV validation only; real-camera MediaPipe accuracy remains unmeasured",
    }
    np.savez_compressed(OUT / "side_view_gates.npz", track=template,
                        threshold_left=np.float32(gates[VIEW_LEFT].threshold),
                        threshold_right=np.float32(gates[VIEW_RIGHT].threshold))
    torch.save(best_state, OUT / "side_view_mt_stgcn.pt")
    (OUT / "side_view_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
