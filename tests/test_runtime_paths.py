from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys_path = str(PROJECT_ROOT / "src")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from ai_trainer.runtime_paths import (
    configured_dataset_path,
    load_dataset_path,
    save_dataset_path,
    user_config_dir,
)


class RuntimePathTests(unittest.TestCase):
    def test_user_config_dir_uses_platform_conventions(self) -> None:
        home = Path("/home/example")
        self.assertEqual(
            user_config_dir(platform_name="linux", environ={}, home=home),
            home / ".config" / "ai-trainer",
        )
        self.assertEqual(
            user_config_dir(platform_name="darwin", environ={}, home=home),
            home / "Library" / "Application Support" / "AI Trainer",
        )
        self.assertEqual(
            user_config_dir(
                platform_name="win32",
                environ={"LOCALAPPDATA": "C:/Users/example/AppData/Local"},
                home=home,
            ),
            Path("C:/Users/example/AppData/Local") / "AI Trainer",
        )

    def test_dataset_setting_round_trip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            config = root / "config" / "settings.json"

            save_dataset_path(dataset, config)

            self.assertEqual(load_dataset_path(config), dataset.resolve())
            with self.assertRaisesRegex(ValueError, "does not exist"):
                save_dataset_path(root / "missing", config)

    def test_environment_dataset_path_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            with patch.dict(os.environ, {"AI_TRAINER_DATASET_PATH": str(dataset)}):
                self.assertEqual(configured_dataset_path(), dataset)


if __name__ == "__main__":
    unittest.main()
