"""실행 모드 선택 다이얼로그: 사용자 모드 / 디버깅 모드."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

MODE_USER = "user"
MODE_DEBUG = "debug"


class ModeSelectDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Trainer · 실행 모드")
        self.setModal(True)
        self.mode: str | None = None

        heading = QLabel("어떤 모드로 실행할까요?")
        heading.setStyleSheet("font-size: 17px; font-weight: 600;")
        heading.setAlignment(Qt.AlignCenter)

        user_btn = QPushButton("사용자 모드")
        user_btn.setMinimumSize(200, 72)
        user_btn.setToolTip("평소 화면만 띄웁니다.")
        debug_btn = QPushButton("디버깅 모드")
        debug_btn.setMinimumSize(200, 72)
        debug_btn.setToolTip("입력→출력까지 스켈레톤 처리 공정 10단계를 별도 창으로 함께 봅니다.")
        for btn in (user_btn, debug_btn):
            btn.setStyleSheet("font-size: 15px;")

        hint = QLabel("다음부터 바로 켜려면:  python scripts/run_game.py --user  또는  --debug")
        hint.setStyleSheet("color: #8a93a3; font-size: 11px;")
        hint.setAlignment(Qt.AlignCenter)

        buttons = QHBoxLayout()
        buttons.addWidget(user_btn)
        buttons.addWidget(debug_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(16)
        layout.addWidget(heading)
        layout.addLayout(buttons)
        layout.addWidget(hint)

        user_btn.clicked.connect(lambda: self._choose(MODE_USER))
        debug_btn.clicked.connect(lambda: self._choose(MODE_DEBUG))
        user_btn.setDefault(True)

    def _choose(self, mode: str) -> None:
        self.mode = mode
        self.accept()


def choose_mode(parent=None) -> str | None:
    """선택한 모드 문자열을 돌려준다. 창을 닫으면 None."""
    dialog = ModeSelectDialog(parent)
    return dialog.mode if dialog.exec_() == QDialog.Accepted else None


__all__ = ["MODE_DEBUG", "MODE_USER", "ModeSelectDialog", "choose_mode"]
