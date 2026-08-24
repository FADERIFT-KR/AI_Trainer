# Online DTW — 후속 정량 검증 TODO

> Online DTW baseline은 승인되어 checkpoint로 보존됨 ([online_dtw.py](../ai_trainer/online_dtw.py),
> 결과: [output/dtw_eval/online_eval_report.json](../output/dtw_eval/online_eval_report.json)).
> 지금 단계에서는 알고리즘을 추가 튜닝하지 않고, 아래 항목만 후속 검증 대상으로 기록한다.

1. **Validation 전체(또는 충분히 큰 표본)로 Online classification 성능 재평가** — 현재는 6개 시퀀스만 스모크 테스트함 (`scripts/test_online_dtw.py`의 `N_TEST_SEQS`를 늘려 재실행).
2. **Online vs Offline prediction agreement rate** — 현재 6개 중 4개 일치(정량화 안 됨, 표본 부족).
3. **Online confusion matrix / class별 precision / recall / F1** — offline과 동일한 형식으로 산출 필요.
4. **Repetition detection 성능 정량화**:
   - start frame error (검출 시작 - GT `start_frame`)
   - end frame error (검출 종료 - GT `end_frame`)
   - temporal IoU (검출 구간 ∩ GT 구간 / 검출 구간 ∪ GT 구간)
   - missed rep(= GT 구간과 전혀 안 겹치는 경우) / false rep(= GT 구간과 안 겹치는 검출) 비율
   - 참고: 현재 스모크 테스트에서 클립 하나당 1~4개의 rep-유사 사이클이 검출됨 — annotation 앞뒤 버퍼 구간에 실제 추가 동작이 있는 것으로 추정되나 확정 검증 필요.
5. **4프레임 latency의 실사용 UX 영향 확인** — 현재는 이론적 지연폭만 계산(`~4/fps`초). 실제 사용자가 피드백 지연을 체감하는지는 실사용 테스트 필요.

## Webcam 파이프라인 관련 (branch 분리로 이전됨)

> 2026-08-19: 웹캠 2D Pose Estimation(MediaPipe 캡처, `pose_estimator.py`/`mediapipe_mapping.py`/
> `run_webcam_pipeline.py`)은 이 브랜치에서 제거하고 `feature/game-ui` 브랜치(ms.choe의 `live_pose`
> 모듈과 병합)로 이전했다. 아래 확인 항목은 그 브랜치에서 계속 추적한다.

6. **실제 카메라/MediaPipe로 2D→3D 결과 시각 검증**:
   - 관절 매핑(MediaPipe 33 -> Common 18)이 실제로 맞는지
   - 2D 정규화가 학습 시(AI Hub 데이터)와 같은 방식으로 동작하는지
   - 좌우 관절이 뒤집히지 않는지 (거울 모드로 프레임을 뒤집어서 화면에 표시하는 경우, pose estimation은 반드시 **뒤집기 전 원본 프레임**에 적용해야 함)
   - occlusion 시 confidence 처리(freeze)가 실제로 부자연스러운 튐을 막아주는지
   - AI Hub 2D CSV와 실제 MediaPipe 출력 사이 domain gap이 lifting 3D 품질에 미치는 영향(MPJPE 재평가 불가하지만 시각적 합리성으로 1차 판단)
