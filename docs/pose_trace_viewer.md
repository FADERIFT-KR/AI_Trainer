# 저장된 3D 진단 로그 보기

게임이나 웹캠을 실행하지 않고 기존 JSONL을 읽는 독립 Qt 뷰어입니다.
입력 파일은 읽기 전용으로 열며 저장·내보내기 기능은 없습니다.

이미지 연결 로그(schema_version 3)는 같은 관측의 원본 이미지, Image Landmarks 오버레이,
World 정면/측면을 함께 표시합니다. [이미지 기록 안내](pose_observation_images.md)를 참고하세요.
과거 로그는 이미지 기록 없음으로 표시하고 기존 좌표 탐색을 유지합니다.

프로젝트 루트 PowerShell:

```powershell
Set-Location C:\AI_Trainer
.\.venv\Scripts\python.exe -B .\scripts\view_pose_trace.py ".\output\diagnostics\squat_pose_20260920_230655_706.jsonl"
```

다른 파일은 마지막 경로를 바꿉니다. 한글·공백이 있는 경로도 따옴표로 감싸면 됩니다.
`--row 100`을 추가하면 0부터 세는 관측 행 100에서 시작합니다.
이전/다음 버튼, 좌우 화살표 또는 관측 행 숫자 입력으로 이동합니다.
창을 닫으면 종료합니다.

## 표시 항목

- World 33관절: 아래 수치 표에 원본 XYZ·visibility를 표시합니다.
- World → 공통 18관절: 기존 `inspect_pose_trace.raw_common()`으로 매핑만 합니다.
- `common3d`: 로그의 필터·저신뢰 관절 유지·Hip 중심 이동 결과 그대로입니다.
- `aligned_frame`: 로그에 남은 마지막 보간 샘플의 정규화 결과 그대로입니다.
- 각 단계는 기존 `SkeletonViews`로 XY 정면·ZY 측면을 나란히 표시합니다.
- 원본 매핑과 common3d는 전체 로그에서 같은 고정 화면 변환을 사용합니다.
  원점 이동을 추가로 보정하지 않으므로 Hip 중심 이동에 따른 위치 차이는 남습니다.
- aligned는 몸 기준 축·무차원 좌표이며 별도의 고정 화면 변환을 사용합니다.
- 모든 투영은 +Y를 화면 위로 그립니다. World 좌표의 실제 위쪽 또는
  카메라의 해부학적 정면을 자동 판별하지 않습니다.
- L/R은 모델 관절 이름이며 화면의 왼쪽/오른쪽을 뜻하지 않습니다.
- 무릎각은 Hip–Knee–Ankle, 고관절각은 Neck–Hip–Knee입니다.
  차이는 필터 후 − 원본이며 길이 단위는 m입니다.
- 오른쪽 Hip–Knee는 같은 관측의 필터 전후 차이와 관측 0 대비 변화도 표시합니다.

## 먼저 볼 관측 찾기

```powershell
.\.venv\Scripts\python.exe -B .\scripts\inspect_pose_trace.py ".\output\diagnostics\squat_pose_20260920_230655_706.jsonl"
```

출력의 각 REP `bottom_frame`을 뷰어의 관측 행 칸에 넣습니다.
이 값은 **0-based JSONL 관측 행**이며 파일 편집기의 줄 번호는 +1입니다.
`sample_index`는 30Hz 처리 번호이므로 대신 입력하면 안 됩니다.
최저점 전후 몇 관측을 이동하며 원본/필터 각도 차이, 오른쪽 Hip–Knee 길이,
frozen3d를 함께 봅니다. 원본과 필터의 픽셀 각도 대신 수치 표의 3D 각도를 비교합니다.

## 제한과 오류 처리

원본 World와 common3d는 같은 관측에 대응하지만 필터에는 과거 관측 영향이 있습니다.
aligned_frame은 같은 행의 마지막 30Hz 보간 샘플로, 관측 timestamp와
정확히 같은 시각이 아닐 수 있습니다. 뷰어는 누락된 중간 좌표·시각을 생성하지 않습니다.
정규화 전후는 참고 비교이며 실제 동시각 비교가 아닙니다.

누락/null/잘못된 형태/비유한 좌표는 해당 단계 전체를 표시 불가로 처리합니다.
0 길이 벡터의 관절각 역시 표시 불가입니다. 손상된 JSON이나 빈 줄은 건너뛰지 않고
파일 줄 번호를 포함한 오류로 중단하여 관측 행 번호가 바뀌지 않게 합니다.
필터·정규화·판정은 재실행하지 않으며 결과 표시는 저장된 값을 읽을 뿐입니다.
정확한 실제 자세와 바닥 접촉은 이 로그만으로 확정할 수 없습니다.

테스트 (임시 디렉터리에만 합성 파일 생성):

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_pose_trace_viewer tests.test_skeleton_panel tests.test_pose_bridge tests.test_pose_diagnostics -v
```
