"""core의 제너릭 UI 뼈대 — 카메라 장치 선택, MediaPipe->Common Skeleton 브릿지,
지터 저감(One-Euro Filter), 스파이크 가드, 관절 오버레이 등 종목 무관 화면 구성요소.

`QMainWindow`/`QStackedWidget` 셸(`app.py`)은 종목별 화면(예: `ai_trainer.squat.game_ui`)을
자식 위젯으로 붙여 쓰는 형태로 재사용한다.
"""
