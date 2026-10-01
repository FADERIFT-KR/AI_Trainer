"""Build leakage-safe fixed-length records directly from AI Hub archives."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ai_trainer.core.s3_mapping.common_skeleton import to_common_skeleton
from ai_trainer.squat.actor_split import OriginSequence, build_actor_split, load_air_squat_sequences
from ai_trainer.squat.aihub_zip import AiHubZip

from .preprocessing import normalize_sequence, smooth_phase_gate
from .schema import LABEL_TO_ID, SequenceMetadata, SequenceRecord


def discover_sequences(dataset_paths: Iterable[str | Path]) -> list[OriginSequence]:
    result: list[OriginSequence] = []
    for index, path in enumerate(dataset_paths):
        result.extend(load_air_squat_sequences(path, origin=f"source{index}"))
    return result


def make_actor_split(
    dataset_paths: Iterable[str | Path], val_ratio: float = 0.2, seed: int = 42
) -> dict[str, str]:
    return build_actor_split(discover_sequences(dataset_paths), val_ratio=val_ratio, seed=seed)


def save_actor_split(actor_map: dict[str, str], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"actor_to_split": dict(sorted(actor_map.items()))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_actor_split(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("actor_to_split", payload)


def load_records(
    dataset_paths: Iterable[str | Path],
    actor_map: dict[str, str],
    split: str,
    labels: set[str] | None = None,
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for source_index, path in enumerate(dataset_paths):
        source = f"source{source_index}"
        with AiHubZip(path) as dataset:
            for sequence in dataset.iter_air_squat_sequences():
                if actor_map.get(sequence.actor) != split:
                    continue
                if labels is not None and sequence.error_type not in labels:
                    continue
                _, coords_26 = dataset.read_3d(sequence)
                normalized, context = normalize_sequence(to_common_skeleton(coords_26))
                records.append(
                    SequenceRecord(
                        coords=normalized,
                        label=sequence.error_type,
                        phase_gate=smooth_phase_gate(normalized),
                        metadata=SequenceMetadata(
                            source=source,
                            actor=sequence.actor,
                            level=sequence.level,
                            repetition=sequence.rep,
                            original_label=sequence.error_type,
                            split=split,
                        ),
                        context=context,
                    )
                )
    return records


class TorchSequenceDataset(Dataset):
    def __init__(self, records: list[SequenceRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        record = self.records[index]
        return {
            "coords": torch.from_numpy(record.coords),
            "phase_gate": torch.from_numpy(record.phase_gate),
            "label": LABEL_TO_ID[record.label],
            "index": index,
        }


def class_weights(records: list[SequenceRecord]) -> torch.Tensor:
    counts = np.bincount([LABEL_TO_ID[record.label] for record in records], minlength=len(LABEL_TO_ID))
    if (counts == 0).any():
        raise ValueError("every class must be represented in the training records")
    return torch.tensor(len(records) / (len(counts) * counts), dtype=torch.float32)
