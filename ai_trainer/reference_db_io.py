"""Reference DB(manifest.json + sequences.npz) 로드 유틸리티."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .features import extract_all_features


def load_reference_db(db_dir: str | Path) -> dict[str, dict[str, list[dict]]]:
    """반환: db[tier][class_label] = [{"feat":..., "bounds":..., "meta":...}, ...]."""
    db_dir = Path(db_dir)
    manifest = json.loads((db_dir / "manifest.json").read_text(encoding="utf-8"))["entries"]
    arrays = np.load(db_dir / "sequences.npz")

    db: dict[str, dict[str, list[dict]]] = {"ground_truth": defaultdict(list), "operational": defaultdict(list)}
    for e in manifest:
        coords = arrays[e["array_key"]]
        feat = extract_all_features(coords)
        db[e["tier"]][e["class_label"]].append({"feat": feat, "bounds": e["phase_boundaries"], "meta": e})
    return db
