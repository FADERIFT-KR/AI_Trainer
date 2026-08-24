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

# CSV/JSON 기반 정상(또는 오류) 스쿼트 스켈레톤 재생 — 웹캠/MediaPipe 미사용
python scripts/play_reference_skeleton.py
```

> ⚠️ 웹캠으로 실제 사용자 자세를 촬영/추적하는 기능(2D Pose Estimation)은 이 브랜치(`feature/dtw-pipeline`)에는
> 없습니다. 해당 기능은 팀원(ms.choe) 브랜치의 `live_pose` 모듈이 담당하며, 두 브랜치를 합친
> `feature/game-ui` 브랜치에서 최종 UI로 통합됩니다. 이 브랜치는 **AI Hub CSV/JSON 데이터 처리,
> 2D→3D Lifting, 정규화, Phase Segmentation, Reference DB, Weighted DTW 비교/채점 엔진**을 담당합니다.
