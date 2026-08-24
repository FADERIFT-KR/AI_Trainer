"""메인 윈도우: 운동 선택 화면 <-> 비교 화면 전환."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QKeyEvent
from PyQt5.QtWidgets import QMainWindow, QStackedWidget

from .screens import CompareScreen, SelectionScreen


class GameWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Trainer · 스쿼트 자세 비교")
        self.resize(1400, 820)

        self.selection_screen = SelectionScreen()
        self.compare_screen = CompareScreen()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.selection_screen)
        self.stack.addWidget(self.compare_screen)
        self.setCentralWidget(self.stack)

        self.selection_screen.start_requested.connect(self._start_compare)
        self.compare_screen.back_requested.connect(self._back_to_selection)

        self.setStyleSheet("QMainWindow { background: #11151d; }")

    def _start_compare(self, class_label: str, medoid_rank: int) -> None:
        self.compare_screen.start(class_label, medoid_rank)
        self.stack.setCurrentWidget(self.compare_screen)

    def _back_to_selection(self) -> None:
        self.stack.setCurrentWidget(self.selection_screen)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            if self.stack.currentWidget() is self.compare_screen:
                self.compare_screen.stop()
                self._back_to_selection()
            else:
                self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.compare_screen.stop()
        event.accept()


__all__ = ["GameWindow"]
