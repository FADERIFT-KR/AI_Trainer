# AI Trainer

소재부품융합공학과 졸업프로젝트 5조 — 웹캠으로 에어스쿼트 자세를 실시간 분석하고, AI Hub 정상 동작 레퍼런스와 비교해 자세 교정 피드백을 제공하는 프로젝트입니다.

이 브랜치(`feature/game-ui`)는 두 작업을 통합합니다.

- **웹캠 2D/3D 스켈레톤 추출** (`ai_trainer/live_pose/`) — MediaPipe Pose(Task API) + PyQt5
- **AI Hub CSV/JSON 기반 정상 자세 레퍼런스 + Weighted DTW 비교/채점 엔진** (`ai_trainer/` 나머지 모듈, `feature/dtw-pipeline` 유래)

목표 UI: 운동 종목 선택 → 좌(웹캠+내 스켈레톤) / 우(정상 레퍼런스 스켈레톤) 2분할 화면 → 실시간 타이밍 동기화 비교 → 자세 정오 판정. (진행 중, 아래 "다음 단계" 참고)

## 개발 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# MediaPipe Pose 모델(최초 1회)
python scripts/download_pose_model.py
```

## 웹캠 2D/3D 스켈레톤만 단독 실행

```bash
python scripts/run_live_pose.py
```

카메라 화면(2D 관절 오버레이)과 MediaPipe 추정 3D world landmark를 PyQt5 창 좌/우에 표시합니다. 카메라 처리는 별도 QThread에서 실행됩니다. 다른 카메라: `--camera 1`.

> 표시되는 3D 좌표는 MediaPipe가 단일 RGB 프레임에서 추정한 골반 중심 좌표이며 깊이 카메라 실측값이 아닙니다. 전신이 잘 잡히도록 카메라에서 2~3m 거리를 두세요.

## AI Hub 정상 자세 레퍼런스 스켈레톤만 단독 재생

```bash
python scripts/play_reference_skeleton.py
```

`output/reference_db/`(AI Hub `3d_points.csv`+`annotation.json`으로 구축된 정상/오류 4클래스 medoid)를 그대로 반복 재생합니다. 원천 영상은 쓰지 않습니다. Reference DB가 없다면 먼저:

```bash
python scripts/build_actor_split.py
python scripts/train_lifting_baseline.py      # 2D->3D lifting 모델
python scripts/build_reference_db.py
```

## Offline / Online Weighted DTW 평가

```bash
python scripts/run_offline_dtw_eval.py
python scripts/test_online_dtw.py
```

## 다음 단계 (통합 UI, 진행 중)

- [ ] `live_pose`의 `PoseObservation`(2D `image_landmarks`) → `ai_trainer` Common Skeleton Mapping → 2D→3D Lifting → `OnlineSquatSession` 연결
- [ ] PyQt5 메인 창에 운동 선택 화면 추가 (현재는 스쿼트만)
- [ ] 우측 패널을 `play_reference_skeleton.py`의 렌더링 로직 기반 QWidget으로 이식, 사용자 진행률(phase/온라인 DTW 정렬)에 맞춰 재생 위치 동기화
- [ ] 실시간 partial distance + rep 종료 시 최종 Weighted DTW 결과(자세 점수/오류 피드백)를 UI에 표시

## 참고 문서

- 전체 아키텍처와 데이터 분석 결과: [`.claude/claude.md`](.claude/claude.md)
- Offline DTW 설계 결정: [`docs/offline_dtw_baseline.md`](docs/offline_dtw_baseline.md)
- 후속 검증 TODO: [`docs/online_dtw_todo.md`](docs/online_dtw_todo.md)
