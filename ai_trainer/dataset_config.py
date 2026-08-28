"""Shared dataset path configuration for AI Trainer."""

from __future__ import annotations

import os
from pathlib import Path

DATASET_PATH_ENV = "AI_TRAINER_DATASET_PATH"
DEFAULT_DATASET_PATH = Path(r"C:\Users\user\Desktop\TP\dataset")


def get_dataset_path() -> Path:
    """Return the configured dataset root, falling back to the local default."""
    configured_path = os.environ.get(DATASET_PATH_ENV)
    return Path(configured_path).expanduser() if configured_path else DEFAULT_DATASET_PATH


DATASET_PATH = get_dataset_path()
