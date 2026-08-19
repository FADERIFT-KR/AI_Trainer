# AI_Trainer

소재부품융합공학과 졸업프로젝트 5조

웹캠으로 스쿼트 자세를 실시간 분석해 자세 교정 피드백을 제공하는 프로젝트입니다.
전체 아키텍처와 데이터 분석 결과는 [`.claude/claude.md`](.claude/claude.md)에 정리되어 있습니다.

## 개발 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 주요 스크립트

```bash
# AI Hub 라벨링 데이터(zip)만으로 스켈레톤 확인 영상 생성 (원천 영상 미사용)
python scripts/visualize_skeleton.py --zip "<TL.zip 경로>" --error-type 정상 --actor CA01 --list
python scripts/visualize_skeleton.py --zip "<TL.zip 경로>" --error-type 정상 --actor CA01 --rep 2

# actor(피험자) 단위 Train/Validation 재분할 + leakage 검사
python scripts/build_actor_split.py

# 정면 카메라 후보 정량 검증 (camera별 2D-3D projection Procrustes 비교)
python scripts/verify_frontal_camera.py

# Offline / Online Weighted DTW 평가 (Reference DB 필요: scripts/build_reference_db.py 먼저 실행)
python scripts/build_reference_db.py
python scripts/run_offline_dtw_eval.py
python scripts/test_online_dtw.py

# Headless 실시간 파이프라인 (Webcam -> Pose -> Lifting -> Online DTW, GUI 없음)
python scripts/run_webcam_pipeline.py --source webcam:0 --viz-out output/webcam_check.mp4
python scripts/run_webcam_pipeline.py --source video:/path/to/clip.mp4
```

> `run_webcam_pipeline.py`는 최초 실행 시 `models/pose_landmarker_lite.task`(MediaPipe Pose 모델,
> ~5.8MB)가 필요합니다. 없으면 아래 명령으로 받으세요:
> ```bash
> mkdir -p models
> curl -L -o models/pose_landmarker_lite.task \
>   "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
> ```
> `--source webcam:0`은 실행하는 터미널 앱에 macOS 카메라 권한이 있어야 합니다
> (시스템 설정 > 개인정보 보호 및 보안 > 카메라). 권한 문제 없이 먼저 테스트하려면
> `--source video:<영상 파일 경로>`를 사용하세요.
