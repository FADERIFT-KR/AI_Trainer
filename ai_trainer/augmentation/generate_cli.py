"""CLI: export validated synthetic errors from held-in normal AI Hub records."""
from __future__ import annotations

import argparse

import torch

from .dataset import load_actor_split, load_records
from .exporter import export_generated_dataset, load_generator
from .schema import NORMAL_LABEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--split", required=True, help="actor split JSON made by train_cli")
    parser.add_argument("--checkpoint", default="models/error_augmenter.pt")
    parser.add_argument("--output", default="output/generated_errors")
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--device")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    generator, profile, _ = load_generator(args.checkpoint, device)
    actor_map = load_actor_split(args.split)
    normal_records = load_records(args.dataset, actor_map, "train", labels={NORMAL_LABEL})
    result = export_generated_dataset(
        generator, profile, normal_records, args.output, variants=args.variants, device=device
    )
    print(result)


if __name__ == "__main__":
    main()
