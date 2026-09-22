# AI Trainer

소재부품융합공학과 졸업프로젝트 5조 — 웹캠으로 에어스쿼트 자세를 실시간 분석하고, AI Hub 정상 동작 레퍼런스와 비교해 자세 교정 피드백을 제공하는 프로젝트입니다.

목표 UI: 운동 종목 선택 → 좌(웹캠+내 스켈레톤) / 우(정상 레퍼런스 스켈레톤) 2분할 화면 → 실시간 타이밍 동기화 비교 → 자세 정오 판정.

## 패키지 구조

종목이 늘어나도 공통 기반을 재사용할 수 있도록 **core(뇌) + 종목별 모듈(몸)** 로 나눠 둡니다.

```
ai_trainer/
├── core/            # 종목 무관 공통 기반
│   ├── live_pose/   # MediaPipe Pose(Task API) 웹캠 2D/3D 스켈레톤 추출
│   ├── game_ui/     # 제너릭 UI 뼈대 (앱 셸, 포즈 브릿지, One-Euro 필터, 스파이크 가드)
│   ├── camera_*.py  # 카메라 장치 선택 / 렌즈 보정
│   └── common_skeleton.py, normalization.py, render.py, dataset_config.py
└── squat/           # 스쿼트 종목 전용 판정 로직
    ├── game_ui/     # 스쿼트 전용 화면 (3시점 촬영 → 비교/판정 → 결과)
    ├── dtw_compare.py, online_dtw.py, two_stage_squat.py, mt_stgcn.py
    ├── view_conditions.py, session_decision.py, phase_*.py
    └── reference_*.py, aihub_zip.py, *_lifting*.py
```

앞으로 푸쉬업·요가 등을 추가할 때는 `ai_trainer/squat/`과 같은 레벨에 새 서브패키지를 만들고, 마찬가지로 `ai_trainer.core`를 그대로 재사용합니다.

## 통합 게임 UI 실행

```bash
python scripts/download_pose_model.py   # 최초 1회
python scripts/build_reference_db.py    # Reference DB가 없다면 먼저 (README 아래쪽 참고)
python scripts/run_game.py
```

스쿼트 9회 판정이 끝난 결과 화면에서 **녹화 영상·분석자료 저장…**을 누르고 저장할 폴더를 선택할 수 있습니다. 판정 도중 오류가 발생해도 오류 화면에서 지금까지 녹화된 자료를 저장할 수 있습니다. 선택한 폴더 아래에 `squat_session_날짜_시간` 폴더가 생성되며, 녹화가 완료된 시점별 영상(`front/left/right.mp4`, 코덱 환경에 따라 `.avi`), 같은 프레임 번호의 관절·판정 기록(`*.jsonl`), 촬영 정보(`*.json`), 최종 판정 또는 오류 정보(`session.json`)가 들어갑니다. 영상은 카운트다운 종료 후부터 녹화한 오버레이 없는 카메라 화면이며, 카메라 보정이 설정되어 있으면 보정된 화면입니다. 오디오는 녹음하지 않습니다. 저장을 선택하지 않고 나가거나 재시도하면 임시 녹화는 삭제됩니다.

나중에 판정 오류나 관절 튐을 확인할 때는 `python scripts/analyze_recorded_session.py "저장된/squat_session_폴더"`를 실행하세요. 반복 종료 시점과 좌·우 무릎의 단일 프레임 튐 후보가 영상 프레임 번호 및 경과 시간과 함께 출력됩니다. 튐 후보는 자동 진단이 아니라 영상을 다시 볼 위치를 찾기 위한 표식입니다. `*.jsonl`의 `video_frame`은 해당 영상의 0부터 시작하는 프레임 번호이고 `elapsed_ms`는 실제 처리 시각입니다. 영상은 고정 FPS로 저장되므로 긴 녹화에서는 재생 시간보다 이 기록의 경과 시간을 우선 참고하세요. 영상·관절 좌표에는 개인 정보가 포함될 수 있으므로 공유 전 동의를 받고 보관 위치를 확인하세요.

저장된 카메라 영상과 기존/개선 스켈레톤을 나란히 확인하려면 `python scripts/replay_recorded_skeleton.py "저장된/squat_session_폴더" --view right --output output/right_corrected.mp4`를 실행하세요. 우측면에서 스쿼트 최저점의 화면상 다리 길이 변화가 촬영 방향 오류로 오인되어 빠지던 프레임은 계속 추적합니다. 화면에 표시하는 3D 좌표는 발목 깊이 튐과 다리 뼈 길이를 먼저 안정화한 뒤, 측면 2D 관절 위치로 보이는 무릎·발목의 앞뒤/위아래 순서를 보정합니다. 이는 **단안 카메라의 표시용 추정**이지 정확한 3D 복원이 아니며, **DTW/ML 판정 입력에는 사용하지 않습니다.**

운동 선택 화면에서 "스쿼트"를 클릭하면 단안 웹캠 한 대로 **정면 3회 → 좌측면 3회 → 우측면 3회**를 순서대로 측정합니다. 여기서 좌측면은 사용자의 왼쪽 면이 카메라를 향한다는 뜻입니다. 각 시점이 끝날 때 다음 카메라 위치가 안내되며, 좌우 관절이 미러링으로 뒤바뀌지 않도록 원본 방향 영상으로 판정합니다.

운동 선택 화면의 카메라 장치 버튼에서 사용할 웹캠을 먼저 고를 수 있습니다. OS가 장치명을 제공하지 않는 환경에서는 `AI_TRAINER_CAMERA_INDEX`의 OpenCV 인덱스가 기본 버튼으로 표시되며, `카메라 목록 새로고침`으로 연결 후 다시 탐색할 수 있습니다. 선택한 장치는 세 시점과 재시도 동안 유지됩니다.

레퍼런스는 선택한 시점으로 투영해 정상 속도로 재생하고, 사용자의 프레임 속도 차이는 phase-aware DTW가 정렬합니다. REP별 시퀀스 모델 판정과 데이터셋 기반 관절 조건이 같은 오류를 지지해야 오류 표가 쌓입니다. 같은 시점 3회 중 2회 이상 같은 오류가 확인되면 오류로 확정하며, 조건만 반복 위반한 경우에는 오류로 단정하지 않고 `판정 불확실`로 처리합니다. 세 시점이 모두 정상이어야 최종 정상입니다.

> ⚠️ 이 앱은 실제 카메라 권한이 있는 터미널에서 직접 실행해야 합니다(코딩 에이전트 샌드박스에서는 카메라를 열 수 없음).

## 개발 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# MediaPipe Pose 모델(최초 1회)
python scripts/download_pose_model.py
```

원본 AI Hub 데이터셋은 기본적으로 프로젝트 폴더와 같은 위치의 `dataset`을 사용합니다. 현재 기본 경로는 `C:\Users\user\Desktop\TP\dataset`입니다. 다른 위치를 사용할 때는 환경변수로 지정할 수 있습니다.

```powershell
$env:AI_TRAINER_DATASET_PATH = "D:\data\crossfit"
python scripts/build_actor_split.py
```

`dataset`, `스쿼트` 또는 `에어스쿼트` 디렉터리를 지정할 수 있으며 기존 `TL.zip`/`VL.zip` 읽기도 계속 지원합니다.

## 데이터셋 기반 시점·관절 조건

`camera0`~`camera7`은 번호를 하드코딩하지 않고 동기화된 2D keypoint와 3D ground truth의 투영 오차, 얼굴 방향, 약한 원근 카메라 방향을 결합해 분류합니다. 현재 데이터셋 전체를 기준으로 한 결과는 `camera1=정면`, `camera0=후면`, `camera7=좌측면`, `camera3=우측면`이며 나머지는 대각 시점입니다.

```bash
python scripts/build_view_conditions.py
```

이 명령은 361개 시퀀스의 세 시점(총 1,083개 view 시퀀스)을 처리해 [`configs/view_condition_thresholds.json`](configs/view_condition_thresholds.json)을 생성합니다. 정상 범위는 훈련 배우의 2.5~97.5 분위수로 정합니다. 오류 조건은 훈련 배우도 다시 `fit/tune` 배우로 분리해 fit에서 임계값을 학습하고 tune에서 규칙을 선택합니다. 기존 검증 배우의 balanced accuracy는 규칙 선택에 사용하지 않고 최종 보고에만 사용해 평가 누수를 막습니다.

- 정면: 무릎-발끝 내측 편위, 좌우 무릎 굽힘 비대칭, 골반·어깨 기울기, 몸통 측굴, 스탠스 폭
- 좌·우측면: 보이는 쪽 무릎·고관절·발목 각도, 몸통 전경, 허벅지 기울기, 고관절-무릎 높이, 무릎 전방 이동, 뒤꿈치 상승·발 기울기
- 시점 확인: 골반 화면 분리 비율의 데이터셋 중앙값(정면 0.351, 좌측 0.0366, 우측 0.0361)을 사용하되 MediaPipe 도메인 차이를 고려한 허용 구간 적용

현재 데이터셋의 세 오류 라벨(발뒤꿈치·엉덩이하방·고관절 오류)은 측면 규칙의 내부 tune 성능이 가장 높아, 오류 확정 책임은 양 측면에 둡니다. 정면 조건은 정면 유지와 좌우 정렬·비대칭 관찰에는 사용하지만, 이 세 오류를 정면의 상관관계만으로 확정하지 않습니다. 책임 시점은 최종 holdout이 아니라 내부 tune 점수의 최고값에서 0.05 이내인 시점만 자동 선택합니다.

선택된 개별 측면 조건의 최종 검증 balanced accuracy는 약 0.74~1.00이고, 각 책임 측면에서 내부 tune으로 고른 최고 규칙은 0.83~1.00입니다. 이는 각 오류와 정상 사이의 **단일 조건 성능**이지 4-class 전체 시스템 정확도가 아닙니다. 따라서 조건만으로 정상을 선언하거나 오류 클래스를 확정하지 않고 시퀀스 모델의 확인·거부 근거로만 사용합니다.

## 웹캠 2D/3D 스켈레톤만 단독 실행

```bash
python scripts/run_live_pose.py
```

카메라 화면(2D 관절 오버레이)과 MediaPipe 추정 3D world landmark를 PyQt5 창 좌/우에 표시합니다. 카메라 처리는 별도 QThread에서 실행됩니다. 다른 카메라: `--camera 1`.

> 표시되는 3D 좌표는 MediaPipe가 단일 RGB 프레임에서 추정한 골반 중심 좌표이며 깊이 카메라 실측값이 아닙니다. 전신이 잘 잡히도록 카메라에서 2~3m 거리를 두세요.

## 카메라 렌즈 보정 (권장)

논문의 발 기울기 기반 회전 보정 대신, 실제 카메라의 초점거리·주점·렌즈 왜곡을 추정하는 ChArUco 보정을 적용합니다. 먼저 보드를 100% 크기로 출력합니다.

```bash
python scripts/calibrate_camera.py --board-output output/charuco_5x7.png --board-only
python scripts/calibrate_camera.py --camera 0
```

두 번째 명령에서 보드를 화면 가장자리와 여러 기울기로 움직이며 20개 장면을 캡처합니다. RMS 오류가 기준(기본 1 px) 안이면 `configs/local_camera_calibration.json`이 생성되고, 같은 카메라·가로세로 비율로 실행하는 게임 UI와 Live Pose에 자동 적용됩니다. 오프라인 영상 분석도 기본적으로 같은 파일을 사용하며, 다른 파일은 `--calibration 경로`로 지정할 수 있습니다.

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

## 3D 스켈레톤 정제

AI Hub 3D 좌표는 로드 시 Spike 제거 → zero-phase Butterworth Jitter 저감 → soft bone-length 안정화를 기본 적용합니다. 원본 CSV는 수정하지 않으며 `read_3d(..., refine=False)`로 감사할 수 있습니다.

```bash
python scripts/audit_3d_skeleton_refinement.py
python scripts/build_reference_db.py
python scripts/run_offline_dtw_eval.py
```

전체 361개 시퀀스에서 jerk RMS 90.9%, 뼈 길이 변동계수 65.3%가 감소했고, 최저점 이동은 평균 0.5 frame, 무릎각 RMS 변화는 0.87°였습니다. 문헌 근거, 과평활 방지 지표와 상세 수치는 [`docs/skeleton_refinement.md`](docs/skeleton_refinement.md)를 참고하세요. Reference medoid는 train actor에서만 선택하며 validation actor와의 교집합을 평가 전에 검사합니다.

## Offline / Online Weighted DTW 평가

```bash
python scripts/run_offline_dtw_eval.py
python scripts/test_online_dtw.py
```

## 참고 문서

- 전체 아키텍처와 데이터 분석 결과: [`.claude/claude.md`](.claude/claude.md)
- Offline DTW 설계 결정: [`docs/offline_dtw_baseline.md`](docs/offline_dtw_baseline.md)
- 후속 검증 TODO: [`docs/online_dtw_todo.md`](docs/online_dtw_todo.md)

# Two-stage squat diagnosis

To build the normal 3D DTW template, calibrate its pass threshold, and train the
multi-task graph error diagnoser, run `python scripts/train_two_stage_squat.py`.
The game uses the generated artifacts automatically. See
[`docs/two_stage_squat.md`](docs/two_stage_squat.md) for evaluation limits and
the weak body-part labels.
