"""AI Trainer core — 카메라 캡처, MediaPipe 포즈 추출, Common Skeleton 매핑, UI 뼈대.

종목(스쿼트/푸쉬업/요가 등)과 무관하게 모든 운동 판정 모듈이 공유하는 하위 계층.
종목별 판정 로직(DTW, 분류기, 레퍼런스 DB 등)은 이 패키지가 아니라 `ai_trainer.squat`
같은 종목별 서브패키지에 둔다.
"""
