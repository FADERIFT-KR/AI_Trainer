#!/usr/bin/env python3
"""Open a saved squat session in the final error-video replay window.

Example:
    python scripts/play_saved_error_replay.py C:\\Users\\user\\Desktop\\TP\\saved\\squat_session_...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

from ai_trainer.game_ui.post_session_replay import build_replay_views  # noqa: E402
from ai_trainer.game_ui.screens import PostSessionReplayDialog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_directory", type=Path, help="saved/squat_session_* directory")
    args = parser.parse_args()
    directory = args.session_directory.expanduser().resolve()
    session_path = directory / "session.json"
    if not session_path.is_file():
        raise SystemExit(f"session.json을 찾을 수 없습니다: {session_path}")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    repetitions = session.get("summary", {}).get("repetitions", [])
    replay_views = build_replay_views(directory, repetitions)
    if not replay_views:
        raise SystemExit("최종 오류로 표시할 반복 또는 재생 가능한 영상이 없습니다.")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv[:1])
    dialog = PostSessionReplayDialog(replay_views)
    dialog.exec_()
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
