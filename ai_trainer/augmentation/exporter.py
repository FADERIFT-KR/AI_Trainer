"""Generate, validate and export auditable NPZ skeleton samples."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import ConditionalErrorGenerator
from .preprocessing import denormalize_sequence
from .projection import orthographic_project
from .rules import RuleProfile
from .schema import ERROR_LABELS, SequenceRecord
from .validator import validate_generated


def load_generator(path: str | Path, device: torch.device) -> tuple[ConditionalErrorGenerator, RuleProfile, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator = ConditionalErrorGenerator().to(device)
    generator.load_state_dict(checkpoint["generator"])
    return generator.eval(), RuleProfile.from_dict(checkpoint["rule_profile"]), checkpoint.get("metrics", {})


@torch.no_grad()
def export_generated_dataset(
    generator: ConditionalErrorGenerator,
    profile: RuleProfile,
    normal_records: list[SequenceRecord],
    output_dir: str | Path,
    *,
    strengths: tuple[float, ...] = (0.75, 1.0, 1.25),
    yaw_degrees: tuple[float, ...] = (-30.0, 0.0, 30.0),
    variants: int = 1,
    device: torch.device | None = None,
) -> dict[str, int]:
    destination = Path(output_dir)
    manifest_path = destination / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {manifest_path}")
    destination.mkdir(parents=True, exist_ok=True)
    device = device or next(generator.parameters()).device
    counts = {"attempted": 0, "accepted": 0, "rejected": 0}
    manifest_rows: list[dict] = []

    for record_index, record in enumerate(normal_records):
        source = torch.from_numpy(record.coords[None]).to(device)
        gate = torch.from_numpy(record.phase_gate[None]).to(device)
        for error_id, label in enumerate(ERROR_LABELS):
            for strength in strengths:
                for variant in range(variants):
                    counts["attempted"] += 1
                    generated = generator(
                        source,
                        torch.tensor([error_id], device=device),
                        torch.tensor([[strength]], device=device),
                        gate,
                    )[0].cpu().numpy()
                    validation = validate_generated(
                        generated, record.coords, record.phase_gate, label, profile
                    )
                    if not validation.accepted:
                        counts["rejected"] += 1
                        continue
                    counts["accepted"] += 1
                    world = (
                        denormalize_sequence(generated, record.context)
                        if record.context is not None else generated
                    )
                    views = np.stack(
                        [orthographic_project(generated, yaw_degrees=yaw) for yaw in yaw_degrees]
                    ).astype(np.float32)
                    stem = f"{record.metadata.actor}_r{record.metadata.repetition}_{error_id}_s{strength:.2f}_v{variant}_{record_index}"
                    file_name = stem.replace(".", "p") + ".npz"
                    np.savez_compressed(
                        destination / file_name,
                        coords_3d_normalized=generated.astype(np.float32),
                        coords_3d_world=world.astype(np.float32),
                        coords_2d_views=views,
                        view_yaw_degrees=np.asarray(yaw_degrees, dtype=np.float32),
                        phase_gate=record.phase_gate,
                    )
                    manifest_rows.append(
                        {
                            "file": file_name,
                            "label": label,
                            "strength": strength,
                            "variant": variant,
                            "source": record.metadata.to_dict(),
                            "validation": validation.to_dict(),
                            "synthetic": True,
                        }
                    )
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    (destination / "summary.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return counts
