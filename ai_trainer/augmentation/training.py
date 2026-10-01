"""Training orchestration for the real-data verifier and conditional generator."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .constraints import total_constraint_loss
from .dataset import TorchSequenceDataset, class_weights
from .model import ConditionalErrorGenerator, ErrorClassifier
from .rules import RuleProfile, feature_profile_loss
from .schema import ERROR_LABELS, LABEL_TO_ID, NORMAL_LABEL, SequenceRecord
from .validator import validate_generated


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    classifier_epochs: int = 80
    generator_epochs: int = 120
    learning_rate: float = 1e-3
    profile_weight: float = 1.0
    constraint_weight: float = 1.0
    semantic_weight: float = 1.0
    identity_weight: float = 0.05


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(
    records: list[SequenceRecord], config: TrainingConfig, device: torch.device
) -> tuple[ErrorClassifier, list[float]]:
    model = ErrorClassifier().to(device)
    loader = DataLoader(TorchSequenceDataset(records), batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights(records).to(device))
    history: list[float] = []
    model.train()
    for _ in range(config.classifier_epochs):
        total, count = 0.0, 0
        for batch in loader:
            coords = batch["coords"].to(device)
            labels = batch["label"].to(device)
            loss = criterion(model(coords), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(coords)
            count += len(coords)
        history.append(total / max(count, 1))
    return model.eval(), history


@torch.no_grad()
def classifier_accuracy(
    model: ErrorClassifier, records: list[SequenceRecord], device: torch.device
) -> float:
    if not records:
        return float("nan")
    loader = DataLoader(TorchSequenceDataset(records), batch_size=64)
    correct = total = 0
    for batch in loader:
        coords = batch["coords"].to(device)
        labels = batch["label"].to(device)
        correct += int((model(coords).argmax(1) == labels).sum())
        total += len(coords)
    return correct / max(total, 1)


def train_generator(
    normal_records: list[SequenceRecord],
    classifier: ErrorClassifier,
    profile: RuleProfile,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[ConditionalErrorGenerator, list[dict[str, float]]]:
    if not normal_records:
        raise ValueError("at least one normal training record is required")
    generator = ConditionalErrorGenerator().to(device)
    loader = DataLoader(TorchSequenceDataset(normal_records), batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    classifier.eval()
    history: list[dict[str, float]] = []

    for _ in range(config.generator_epochs):
        sums = {name: 0.0 for name in ("total", "semantic", "profile", "constraints", "identity")}
        count = 0
        for batch in loader:
            source = batch["coords"].to(device)
            gate = batch["phase_gate"].to(device)
            error_id = torch.randint(len(ERROR_LABELS), (len(source),), device=device)
            strength = torch.ones(len(source), 1, device=device)
            target_labels = [ERROR_LABELS[index] for index in error_id.tolist()]
            generated = generator(source, error_id, strength, gate)

            target_ids = error_id + LABEL_TO_ID[ERROR_LABELS[0]]
            semantic = nn.functional.cross_entropy(classifier(generated), target_ids)
            profile_loss = feature_profile_loss(generated, gate, target_labels, profile)
            constraints, _ = total_constraint_loss(generated, source, gate)
            identity = (generated - source).square().mean()
            loss = (
                config.semantic_weight * semantic
                + config.profile_weight * profile_loss
                + config.constraint_weight * constraints
                + config.identity_weight * identity
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            optimizer.step()

            values = {
                "total": loss, "semantic": semantic, "profile": profile_loss,
                "constraints": constraints, "identity": identity,
            }
            for name, value in values.items():
                sums[name] += float(value.detach()) * len(source)
            count += len(source)
        history.append({name: value / max(count, 1) for name, value in sums.items()})
    return generator.eval(), history


@torch.no_grad()
def generated_acceptance_rate(
    generator: ConditionalErrorGenerator,
    records: list[SequenceRecord],
    profile: RuleProfile,
    device: torch.device,
) -> float:
    normal = [record for record in records if record.label == NORMAL_LABEL]
    accepted = total = 0
    for record in normal:
        source = torch.from_numpy(record.coords[None]).to(device)
        gate = torch.from_numpy(record.phase_gate[None]).to(device)
        for error_id, label in enumerate(ERROR_LABELS):
            generated = generator(
                source,
                torch.tensor([error_id], device=device),
                torch.ones(1, 1, device=device),
                gate,
            )[0].cpu().numpy()
            report = validate_generated(generated, record.coords, record.phase_gate, label, profile)
            accepted += int(report.accepted)
            total += 1
    return accepted / max(total, 1)


def train_all(
    train_records: list[SequenceRecord],
    validation_records: list[SequenceRecord],
    output_path: str | Path,
    config: TrainingConfig | None = None,
    device_name: str | None = None,
) -> dict:
    cfg = config or TrainingConfig()
    _seed_everything(cfg.seed)
    device = _device(device_name)
    profile = RuleProfile.fit(train_records)
    classifier, classifier_history = train_classifier(train_records, cfg, device)
    normal_records = [record for record in train_records if record.label == NORMAL_LABEL]
    generator, generator_history = train_generator(normal_records, classifier, profile, cfg, device)
    metrics = {
        "classifier_train_accuracy": classifier_accuracy(classifier, train_records, device),
        "classifier_validation_accuracy": classifier_accuracy(classifier, validation_records, device),
        "generated_validation_acceptance_rate": generated_acceptance_rate(
            generator, validation_records, profile, device
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "classifier": classifier.state_dict(),
            "rule_profile": profile.to_dict(),
            "training_config": asdict(cfg),
            "metrics": metrics,
        },
        destination,
    )
    report_path = destination.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "classifier_loss": classifier_history,
                "generator_loss": generator_history,
                "rule_profile": profile.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics
