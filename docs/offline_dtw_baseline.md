# Offline Weighted DTW Baseline — 결정 사항 기록

> `feature/dtw-pipeline` 브랜치. Offline 평가 결과: [output/dtw_eval/offline_eval_report.json](../output/dtw_eval/offline_eval_report.json)

## 1. Weight Profile

- **결정**: 당분간 `E_full_uniform`(11개 feature 균등 가중치)을 기본값으로 사용한다.
- **근거**: Validation accuracy — uniform 0.726 vs custom(D, class_overrides) 0.685.
- `D_full_weighted`(class_overrides 포함, `configs/dtw_feature_weights.json`)는 **폐기하지 않고 "초기 가설 기반 configuration"으로 보존**. Validation 결과에 맞춘 수동 튜닝은 하지 않았다 — 전체 파이프라인(Online DTW까지) 완성 후 별도 ablation 단계에서 재검토 예정.
- `configs/dtw_feature_weights.json`의 `default_profile` 필드로 코드에서 참조한다 (하드코딩 금지 원칙 유지).

## 2. 발뒤꿈치오류 Recall (0.5) — baseline으로 기록, 지금 최적화하지 않음

Confusion matrix (operational tier / min_distance, true=발뒤꿈치오류 행):

| true\pred | 정상 | 발뒤꿈치오류 | 엉덩이하방오류 | 고관절오류 |
|---|---|---|---|---|
| 발뒤꿈치오류 (16) | **8** | 8 | 0 | 0 |

- 오분류는 **100% "정상"으로만** 쏠림 (다른 오류 클래스로는 전혀 섞이지 않음) — 즉 현재 feature/weight 구성이 발뒤꿈치오류를 "오류 없음"과 잘 구분하지 못하는 것이지, 다른 오류와 혼동하는 문제는 아님.
- 향후 개선 후보 feature (기록만, 지금 구현 안 함):
  - heel height (현재도 포함되어 있으나 가중치/판별력 부족 가능성)
  - ankle angle
  - heel/toe relative geometry (현재 knee_toe_alignment만 있고 heel-toe 관계는 없음 — 신규 feature 후보)
  - foot trajectory (현재는 정적 heel height뿐, 궤적 자체는 미포함)
  - 최저점 및 상승 초기 phase의 heel 변화량 (phase-local 미분 feature 후보)

## 3. Reference Tier

- **실시간(Online) 비교의 기본 Reference: Operational Reference.** 사용자 입력과 Reference 모두 동일한 2D→3D lifting 경로를 통과해 systematic lifting error / coordinate-domain gap을 줄인다 (Offline 평가에서 GT 대비 accuracy 0.685→0.740, binary 0.795→0.836로 실측 확인).
- **Ground Truth Reference는 계속 유지**하며 용도를 다음으로 한정: (1) 3D lifting 모델 정확도 평가(MPJPE), (2) Reference 품질 검증, (3) biomechanical feature 분석(오류유형별 특징), (4) Operational Reference와의 성능 비교 기준선.

## Baseline 수치 요약 (재현 기준점)

- Lifting 모델: `output/lifting_baseline/model_best.pt`, val MPJPE 48.72mm, T=9, 파라미터 125,046개
- Reference DB: `output/reference_db/` (4클래스 × 4 medoid × 2 tier, train-actor만 사용, val actor와 겹침 없음 확인)
- Offline DTW (Operational tier, min_distance, uniform weight 적용 전 D 기준 수치는 위 표 참고): accuracy 0.740, binary(정상 vs 오류) 0.836
