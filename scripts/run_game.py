#!/usr/bin/env python3
"""게임형 통합 UI 실행: 운동 선택 -> (좌)웹캠+내 스켈레톤 / (우)정상 레퍼런스 2분할 비교.

웹캠 2D 추정은 ms.choe의 live_pose(MediaPipe Task API), 정상 레퍼런스와
Weighted DTW 채점은 이 브랜치(feature/dtw-pipeline 유래)의 AI Hub 기반 파이프라인을 사용한다.

⚠️ 이 코딩 에이전트(샌드박스) 프로세스에서 실행하면 macOS 카메라 권한을 받을 수 없다.
카메라 권한이 있는 실제 터미널(VS Code 통합 터미널 등)에서 사용자가 직접 실행해야 한다.

사전 준비:
    python scripts/download_pose_model.py     # MediaPipe pose 모델
    python scripts/build_reference_db.py       # (없다면) Reference DB 구축

사용:
    python scripts/run_game.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from ai_trainer.game_ui.app import GameWindow  # noqa: E402


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("AI Trainer")
    window = GameWindow()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
