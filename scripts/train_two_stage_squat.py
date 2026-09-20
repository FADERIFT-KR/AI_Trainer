#!/usr/bin/env python3
"""Build a normal DTW template, calibrate its gate, and train MT-ST-GCN.

The existing actor split is respected.  Within training actors, held-out
calibration actors set the DTW threshold and select the graph checkpoint;
outer validation actors are read only for the final report.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from ai_trainer.actor_split import load_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.dataset_config import DATASET_PATH  # noqa: E402
from ai_trainer.lifting_dataset import load_actor_split  # noqa: E402
from ai_trainer.mt_stgcn import MultiTaskSTGCN  # noqa: E402
from ai_trainer.reference_pipeline import build_ground_truth_reference  # noqa: E402
from ai_trainer.two_stage_squat import (  # noqa: E402
    ERROR_CLASSES, PART_TARGETS, NormalTemplateGate, build_normal_template, choose_threshold,
)

SEED = 42
OUT = ROOT / "output" / "two_stage_squat"
EPOCHS = 35
BATCH_SIZE = 16


def time_warp(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Monotone stochastic time perturbation; keeps first/last pose fixed."""
    if rng.random() > 0.5:
        return x
    t = len(x)
    knots = np.linspace(0, t - 1, 7)
    increments = np.exp(rng.normal(0.0, 0.20, len(knots) - 1))
    warped_knots = np.r_[0.0, np.cumsum(increments)]
    warped_knots *= (t - 1) / warped_knots[-1]
    source_t = np.interp(np.arange(t), knots, warped_knots)
    flat = x.reshape(t, -1)
    return np.stack([np.interp(source_t, np.arange(t), flat[:, c]) for c in range(flat.shape[1])], axis=1).reshape(x.shape).astype(np.float32)


def metrics(labels: np.ndarray, predicted: np.ndarray) -> dict:
    normal = labels == "정상"
    passed = predicted == "정상"
    return {
        "n": int(len(labels)),
        "binary_accuracy": float(np.mean(normal == passed)),
        "normal_recall": float(np.mean(passed[normal])) if normal.any() else None,
        "error_false_pass_rate": float(np.mean(passed[~normal])) if (~normal).any() else None,
        "four_class_accuracy": float(np.mean(labels == predicted)),
    }


def run() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(SEED)
    split = load_actor_split(ROOT / "configs" / "actor_split.json")
    train_actors = sorted(a for a, side in split.items() if side == "train")
    rng.shuffle(train_actors)
    n_cal = max(4, round(0.22 * len(train_actors)))
    calibration_actors = set(train_actors[:n_cal])
    fit_actors = set(train_actors[n_cal:])
    records = []
    with AiHubZip(DATASET_PATH) as dataset:
        for os_ in load_air_squat_sequences(DATASET_PATH):
            if os_.seq.actor not in split:
                continue
            ref = build_ground_truth_reference(dataset, os_.seq, os_.origin)
            if ref is not None and len(ref.coords) >= 10:
                records.append({"actor": os_.seq.actor, "label": os_.seq.error_type, "track": ref.coords})
    normal_tracks = [r["track"] for r in records if r["actor"] in fit_actors and r["label"] == "정상"]
    template = build_normal_template(normal_tracks)
    temporary_gate = NormalTemplateGate(template, 1.0)
    cal = [r for r in records if r["actor"] in calibration_actors]
    distances = np.array([temporary_gate.assess(r["track"]).distance for r in cal])
    threshold, threshold_stats = choose_threshold(distances, np.array([r["label"] == "정상" for r in cal]))
    gate = NormalTemplateGate(template, threshold)

    # Cache path-aligned tensors. Only error examples supervise the graph heads.
    fit_errors = [r for r in records if r["actor"] in fit_actors and r["label"] in ERROR_CLASSES]
    cal_errors = [r for r in cal if r["label"] in ERROR_CLASSES]
    x_fit = np.stack([gate.assess(r["track"]).aligned for r in fit_errors])
    y_fit = np.array([ERROR_CLASSES.index(r["label"]) for r in fit_errors], dtype=np.int64)
    p_fit = np.array([PART_TARGETS[r["label"]] for r in fit_errors], dtype=np.float32)
    x_cal = np.stack([gate.assess(r["track"]).aligned for r in cal_errors])
    y_cal = np.array([ERROR_CLASSES.index(r["label"]) for r in cal_errors], dtype=np.int64)
    weights = len(y_fit) / (len(ERROR_CLASSES) * np.maximum(np.bincount(y_fit, minlength=len(ERROR_CLASSES)), 1))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    bce = nn.BCEWithLogitsLoss()
    model = MultiTaskSTGCN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best_acc = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        for indices in np.array_split(rng.permutation(len(x_fit)), max(1, int(np.ceil(len(x_fit) / BATCH_SIZE)))):
            augmented = np.stack([time_warp(x_fit[i], rng) for i in indices])
            batch = torch.from_numpy(augmented.transpose(0, 3, 1, 2))
            error_logits, part_logits = model(batch)
            loss = ce(error_logits, torch.from_numpy(y_fit[indices])) + 0.25 * bce(part_logits, torch.from_numpy(p_fit[indices]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(x_cal.transpose(0, 3, 1, 2)))
            acc = float(np.mean(logits.argmax(1).numpy() == y_cal))
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch + 1
            best_state = {name: value.cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    outer = [r for r in records if split[r["actor"]] == "val"]
    predicted = []
    part_predictions = []
    with torch.no_grad():
        for record in outer:
            gate_result = gate.assess(record["track"])
            if gate_result.passed:
                predicted.append("정상")
                continue
            inp = torch.from_numpy(gate_result.aligned.transpose(2, 0, 1)[None])
            error_logits, part_logits = model(inp)
            predicted.append(ERROR_CLASSES[int(error_logits.argmax(1))])
            part_predictions.append({"actor": record["actor"], "label": record["label"],
                                     "probabilities": torch.sigmoid(part_logits[0]).numpy().round(3).tolist()})
    val_labels = np.array([r["label"] for r in outer])
    val_predictions = np.array(predicted)
    report = {
        "source": str(DATASET_PATH),
        "template_length": int(len(template)),
        "n_template_tracks": len(normal_tracks),
        "fit_actors": sorted(fit_actors),
        "calibration_actors": sorted(calibration_actors),
        "outer_validation_actors": sorted({r["actor"] for r in outer}),
        "threshold": threshold,
        "calibration_metrics": threshold_stats,
        "graph_best_epoch": best_epoch,
        "graph_calibration_error_accuracy": best_acc,
        "outer_validation": metrics(val_labels, val_predictions),
        "outer_confusion": {actual: {pred: int(np.sum((val_labels == actual) & (val_predictions == pred)))
                                    for pred in ("정상", *ERROR_CLASSES)} for actual in ("정상", *ERROR_CLASSES)},
        "part_head_label_source": "weak mapping from error class, not independently annotated anatomy",
        "validation_usage_note": "The same outer actor split informed a lower-body distance-weight revision during development; treat this as exploratory validation, not a sealed final test.",
        "part_predictions": part_predictions,
        "caveat": "3D CSV validation only; webcam domain and 2D MediaPipe world landmarks are not validated here",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "normal_template.npz", track=template, threshold=np.float32(threshold))
    torch.save(best_state, OUT / "mt_stgcn.pt")
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("part_predictions", "outer_confusion")}, ensure_ascii=False, indent=2))
    print("confusion:", json.dumps(report["outer_confusion"], ensure_ascii=False))


if __name__ == "__main__":
    run()
