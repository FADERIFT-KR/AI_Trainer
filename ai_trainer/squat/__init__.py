"""AI Trainer squat — 스쿼트 종목 전용 판정 로직.

AI Hub CSV/JSON 기반 정상 레퍼런스 + Weighted DTW/2단계(NormalTemplateGate+MT-ST-GCN)
분류기 + 시점별 규칙 조건(view_conditions)을 통해 스쿼트 자세를 판정한다. 카메라 캡처·
MediaPipe 포즈 추출·제너릭 UI 뼈대는 `ai_trainer.core`를 그대로 가져다 쓴다.

이후 다른 종목(푸쉬업/요가 등)을 추가할 때는 이 패키지와 같은 레벨에 새 서브패키지를
만들고, 마찬가지로 `ai_trainer.core`를 재사용한다.
"""
