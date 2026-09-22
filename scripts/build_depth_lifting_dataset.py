#!/usr/bin/env python3
"""Derive 27-frame samples from the existing actor-separated lifting dataset.

The old NPZ records the centre 2D frame, target 3D frame and sequence metadata
for each (stride-two) window.  Grouping those records reconstructs a longer
motion context without moving actors across train/validation splits.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ai_trainer.squat.depth_lifting import TEMPORAL_WINDOW  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "output" / "lifting_dataset"
OUT_DIR = ROOT / "output" / "depth_lifting_dataset"


def _sequence_keys(data: np.lib.npyio.NpzFile) -> np.ndarray:
    # origin identifies the source archive; the other fields identify one take.
    return np.char.add(
        np.char.add(np.char.add(np.char.add(data["origin"].astype(str), "|"), data["actor"].astype(str)), "|"),
        np.char.add(np.char.add(np.char.add(data["level"].astype(str), "|"), data["error_type"].astype(str)),
                    np.char.add("|", data["rep"].astype(str))),
    )


def make_long_context(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    x_center = data["x_norm"][:, data["x_norm"].shape[1] // 2].astype(np.float32)
    y = data["y_norm"].astype(np.float32)
    centres = data["center_frame"].astype(int)
    keys = _sequence_keys(data)
    half = TEMPORAL_WINDOW // 2
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[int] = []
    for key in np.unique(keys):
        indices = np.flatnonzero(keys == key)
        indices = indices[np.argsort(centres[indices])]
        if len(indices) < 2:
            continue
        for pos, index in enumerate(indices):
            take = np.clip(np.arange(pos - half, pos + half + 1), 0, len(indices) - 1)
            windows.append(x_center[indices[take]])
            targets.append(y[index])
            metadata.append(index)
    if not windows:
        raise ValueError("No temporal lifting samples could be reconstructed")
    selected = np.asarray(metadata, dtype=np.int64)
    result = {"x_norm": np.stack(windows), "y_norm": np.stack(targets)}
    # Keep identity fields for auditability (not used by the model).
    for name in ("origin", "actor", "level", "error_type", "rep", "center_frame"):
        result[name] = data[name][selected]
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        with np.load(SOURCE_DIR / f"{split}.npz") as source:
            derived = make_long_context(source)
        path = OUT_DIR / f"{split}.npz"
        np.savez_compressed(path, **derived)
        print(f"{split}: {derived['x_norm'].shape} -> {derived['y_norm'].shape}; {path}")


if __name__ == "__main__":
    main()
