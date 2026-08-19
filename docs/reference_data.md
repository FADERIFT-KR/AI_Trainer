# AI Hub 에어스쿼트 reference data

## 1. 원본 범위

- 데이터셋: AI Hub `크로스핏 동작 데이터`
- 데이터셋 ID/버전: `71422` / `1.1`
- 공개 페이지: <https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71422>
- 선별 라벨: 통상 `motion_category3 = 에어 스쿼트`, `motion_category4 = 정상`
- 입력: annotation JSON과 26관절 3D keypoint CSV
- 제외: 2D CSV, 프런트/오버헤드 스쿼트, 발뒤꿈치·엉덩이 하방·고관절 오류 라벨

공식 페이지에는 한 동작을 8개 시점에서 30 FPS로 촬영하고, 2D keypoint 8개와 3D keypoint 1개를 제공한다고 설명되어 있습니다. 공개 설명만으로 실제 압축파일의 JSON↔3D CSV 관계를 확정할 수 없으므로 빌더는 모호한 파일을 임의로 연결하지 않습니다.

## 2. 입력 준비와 pairing

AI Hub에서 본인 확인과 이용 목적 등록 후 데이터셋을 다운로드합니다. 전체 공개 파일은 약 1.44 TB이지만 이 빌더에는 AVI/JPG가 필요하지 않습니다. 파일 목록에서 `Training/02.라벨링데이터/TL.zip`(3,136,220,431 bytes)과 `Validation/02.라벨링데이터/VL.zip`(386,509,510 bytes)만 선택하면 됩니다. 원본은 `data/raw/` 아래에 두며 Git에 포함하지 않습니다. AI Hub shell을 쓸 경우 페이지의 `dataSetSn=71422`를 shell의 인증 후 발급되는 `datasetkey`와 같다고 가정하지 마세요.

자동 pairing은 다음 조건을 모두 사용합니다.

1. `_z` 열로 3D CSV인지 판별
2. JSON 파일명, `video_name`, `video_path`와 CSV 이름 비교
3. 가까운 디렉터리 구조 비교
4. 최고 후보가 단독으로 충분히 확실할 때만 선택

다운로드 구조에서 같은 이름이 반복되면 `configs/aihub_air_squat_pairs.example.csv` 형식으로 직접 연결합니다. 경로는 모두 `--input` 기준 상대 경로여야 하며 입력 루트 밖의 파일은 받지 않습니다. `annotation_index`는 0부터 시작하며 비워 두면 JSON 안의 모든 정상 에어스쿼트 구간을 사용합니다.

여러 카메라 JSON이 같은 3D CSV와 같은 프레임 구간을 가리켜도 `(3D CSV, start row, end row)`를 기준으로 한 번만 집계합니다.

## 3. 처리 규칙

### 스켈레톤

AI Hub 26개 관절 중 MediaPipe Pose와 직접 비교 가능한 19개를 유지합니다.

```text
Nose, Neck,
L/R Shoulder, Elbow, Wrist,
Hip, L/R Hip, Knee, Ankle, Heel, BigToe
```

MediaPipe에서 `Neck`은 11·12번 어깨의 중점, `Hip`은 23·24번 골반의 중점이며, `BigToe`는 31·32번 foot index에 대응합니다.

### 공간 정규화

1. 매 프레임 좌우 골반 중점을 원점으로 이동
2. 시퀀스의 골반 중점→목 거리 중앙값으로 나누어 체형 크기 제거
3. 시퀀스 앞·뒤 준비 자세에서 좌우 골반축과 골반→목 축을 구함
4. 하나의 고정된 오른손 좌표계를 전체 시퀀스에 적용

프레임마다 방향을 다시 맞추지 않으므로 스쿼트 중 몸통 기울기는 보존됩니다.

### feature

- 좌우 무릎·고관절·발목 각도
- 몸통의 전후 기울기
- 좌우 무릎/고관절 각도 비대칭
- 무릎과 발끝의 좌우 tracking 차이
- 골반-발목 높이, 스탠스 폭, 뒤꿈치 높이 비대칭

### 반복 분할과 위상 정렬

JSON frame 구간 안에서 좌우 무릎 굴곡 신호를 사용해 `준비 → 최저점 → 준비`가 완성된 반복만 찾습니다. 기본값은 최소 15 frame, 무릎 굴곡 변화 25° 이상, 유효 관절 비율 95% 이상입니다.

각 반복의 하강과 상승을 각각 51점으로 보간하고 최저점 한 점을 공유하여 총 101점으로 만듭니다. 따라서 최저점은 항상 index `50`이고 서로 다른 수행 속도를 직접 비교할 수 있습니다.

## 4. 산출물 계약

### `reference.npz`

- `positions_median/q10/q90`: `[101, 19, 3]`
- `features_median/q10/q90`: `[101, 14]`
- `representative_positions/features`: 실제 정상 반복 중 중앙 template에 가장 가까운 반복
- `phase`: `0=descent`, `1=bottom`, `2=ascent`
- `phase_progress`, `bottom_index`, `joint_names`, `feature_names`

### `repetitions.npz`

정규화된 개별 반복, feature, QC 기반 accepted mask와 template distance를 담습니다. `--no-repetitions`로 생성을 끌 수 있습니다.

### `reference_preview.png`

빌드 마지막 단계에서 `positions_median`의 준비(`0`), 최저점(`bottom_index`), 복귀(`100`) 자세를 같은 축 범위의 3D 패널로 저장합니다. 파란색은 왼쪽, 주황색은 오른쪽 관절이며 원본 좌표 `(x, y, z)`를 화면 `(x, z, y)`로 표시해 `y`가 위쪽을 향합니다. 자동 플롯이 필요 없으면 `--no-plot`을 사용합니다.

### `manifest.csv`

각 반복의 원본 상대 경로, annotation/row/frame 범위, 굴곡 범위, 유효 비율, 입력 SHA-256, outlier 판정과 거리를 기록합니다. actor의 키·성별·나이는 저장하지 않습니다.

### `metadata.json` / `build_report.json`

데이터셋 출처·버전·이용정책, 빌드 설정, joint/feature 스키마, 처리 개수와 pairing 실패 사유를 기록합니다.

## 5. 실행 예

```powershell
python scripts/build_air_squat_reference.py `
  --input data/raw/aihub_crossfit `
  --pairs-manifest data/raw/aihub_crossfit/pairs.csv `
  --output data/reference/air_squat `
  --target-frames 101 `
  --min-repetitions 3
```

기존 결과를 의도적으로 교체할 때만 `--overwrite`를 추가합니다.

## 6. 해석과 이용 제한

이 reference는 AI Hub가 `정상`으로 라벨한 에어스쿼트의 통계적 대표값이며 의료적 진단 기준이 아닙니다. 사용자 피드백 임계값은 별도의 검증 집단으로 보정해야 합니다.

AI Hub 이용정책은 사업결과 출처 표시, 승인 없는 제3자 열람·제공 금지, 국외 이용·반출 시 별도 합의 등을 규정합니다. 원본과 파생 결과를 공개 저장소나 배포 패키지에 넣지 말고 실제 서비스 이용 방식은 필요 시 한국지능정보사회진흥원에 확인하세요.
