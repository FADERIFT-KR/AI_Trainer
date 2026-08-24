"""Run the live camera pose window without installing this source tree."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.live_pose.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(project_root=PROJECT_ROOT))
