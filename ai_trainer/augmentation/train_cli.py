"""CLI: train the minimal AI Hub-only conditional error augmenter."""
from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import load_records, make_actor_split, save_actor_split
from .training import TrainingConfig, train_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="AI Hub TL/VL zip or extracted directory; repeatable")
    parser.add_argument("--output", default="models/error_augmenter.pt")
    parser.add_argument("--split-output", default="output/error_augmentation_actor_split.json")
    parser.add_argument("--classifier-epochs", type=int, default=80)
    parser.add_argument("--generator-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()

    actor_map = make_actor_split(args.dataset, seed=args.seed)
    save_actor_split(actor_map, args.split_output)
    train_records = load_records(args.dataset, actor_map, "train")
    validation_records = load_records(args.dataset, actor_map, "val")
    config = TrainingConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        classifier_epochs=args.classifier_epochs,
        generator_epochs=args.generator_epochs,
    )
    metrics = train_all(train_records, validation_records, Path(args.output), config, args.device)
    print(metrics)


if __name__ == "__main__":
    main()
