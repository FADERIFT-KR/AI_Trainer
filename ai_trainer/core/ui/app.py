"""메인 윈도우: 운동 선택 화면 <-> 비교 화면 전환.

`debug=True`면 파이프라인 공정별 스켈레톤을 보여주는 디버그 창(ai_trainer.debug)을
메인 창 옆에 함께 띄운다. 사용자 모드에서는 디버그 관련 객체를 전혀 만들지 않는다.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QKeyEvent
from PyQt5.QtWidgets import QMainWindow, QStackedWidget

from ai_trainer.squat.game_ui.screens import CompareScreen, SelectionScreen


class GameWindow(QMainWindow):
    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        self.debug_window = None
        self._debug_tap = None
        if debug:
            from ai_trainer.debug.debug_window import DebugWindow
            from ai_trainer.debug.stages import StageTap

            self._debug_tap = StageTap()
            self.debug_window = DebugWindow()
            self._debug_tap.frame_ready.connect(self.debug_window.on_frame)

        title = "AI Trainer · 스쿼트 자세 비교"
        self.setWindowTitle(title + (" [디버깅 모드]" if debug else ""))
        self.resize(1400, 820)

        self.selection_screen = SelectionScreen()
        self.compare_screen = CompareScreen(debug_tap=self._debug_tap)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.selection_screen)
        self.stack.addWidget(self.compare_screen)
        self.setCentralWidget(self.stack)

        self.selection_screen.start_requested.connect(self._start_compare)
        self.compare_screen.back_requested.connect(self._back_to_selection)

        self.setStyleSheet("QMainWindow { background: #11151d; }")

    def show(self) -> None:  # noqa: D102
        super().show()
        if self.debug_window is None:
            return
        # 두 창이 겹쳐 디버그 창이 뒤에 숨지 않도록, 화면에 위아래로 나눠 배치한다.
        screen = self.screen().availableGeometry()
        debug_h = min(self.debug_window.height(), int(screen.height() * 0.46))
        main_h = min(self.height(), screen.height() - debug_h - 60)
        self.setGeometry(screen.x() + 20, screen.y() + 20,
                         min(self.width(), screen.width() - 40), main_h)
        self.debug_window.setGeometry(
            screen.x() + 20, screen.y() + main_h + 50,
            min(self.debug_window.width(), screen.width() - 40), debug_h)
        self.debug_window.show()
        self.debug_window.raise_()
        self.debug_window.activateWindow()

    def _start_compare(self, class_label: str, medoid_rank: int, camera_index: int) -> None:
        self.compare_screen.start(class_label, medoid_rank, camera_index)
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
        if self.debug_window is not None:
            self.debug_window.close()
        event.accept()


__all__ = ["GameWindow"]
