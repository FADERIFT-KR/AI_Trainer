"""Portable, per-user paths for AI Trainer settings and generated files."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path


APP_DIRECTORY_NAME = "AI Trainer"
SETTINGS_FILE_NAME = "settings.json"
DATASET_ENVIRONMENT_VARIABLE = "AI_TRAINER_DATASET_PATH"


def user_config_dir(
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return a writable, OS-standard user configuration directory.

    The optional arguments keep platform choice deterministic in tests and
    avoid forcing callers to monkeypatch global process state.
    """

    platform_name = sys.platform if platform_name is None else platform_name
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    override = environ.get("AI_TRAINER_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if platform_name.startswith("win"):
        base = environ.get("LOCALAPPDATA") or environ.get("APPDATA")
        return (Path(base) if base else home / "AppData" / "Local") / APP_DIRECTORY_NAME
    if platform_name == "darwin":
        return home / "Library" / "Application Support" / APP_DIRECTORY_NAME
    base = environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else home / ".config") / "ai-trainer"


def settings_path() -> Path:
    return user_config_dir() / SETTINGS_FILE_NAME


def load_dataset_path(path: Path | None = None) -> Path | None:
    """Read the saved dataset directory, returning ``None`` for bad settings."""

    candidate = settings_path() if path is None else Path(path)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        value = payload.get("dataset_path")
        return Path(value).expanduser() if isinstance(value, str) and value else None
    except (OSError, ValueError):
        return None


def configured_dataset_path() -> Path | None:
    """Return an environment override first, then the persisted selection."""

    override = os.environ.get(DATASET_ENVIRONMENT_VARIABLE)
    return Path(override).expanduser() if override else load_dataset_path()


def save_dataset_path(dataset: Path, path: Path | None = None) -> Path:
    """Atomically persist a user-selected, existing dataset directory."""

    dataset = Path(dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    destination = settings_path() if path is None else Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump({"dataset_path": str(dataset)}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def default_model_path(project_root: Path | None = None) -> Path:
    """Prefer a bundled model, otherwise use a writable per-user model path."""

    bundled = Path(project_root) / "models" / "pose_landmarker_lite.task" if project_root else None
    if bundled is not None and bundled.is_file():
        return bundled
    return user_config_dir() / "models" / "pose_landmarker_lite.task"


def reference_output_dir() -> Path:
    """Location for generated reference artifacts when the app is installed."""

    return user_config_dir() / "reference" / "air_squat"


__all__ = [
    "DATASET_ENVIRONMENT_VARIABLE",
    "configured_dataset_path",
    "default_model_path",
    "load_dataset_path",
    "reference_output_dir",
    "save_dataset_path",
    "settings_path",
    "user_config_dir",
]
