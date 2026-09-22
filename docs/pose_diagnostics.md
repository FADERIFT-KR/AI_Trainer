# 스쿼트 3D / 발뒤꿈치 판정 진단

## 관측 이미지 연결 기록 (schema_version 3)

현재 `--pose-trace`는 JSONL과 함께 **사람의 모습이 포함된 로컬 PNG 이미지**도 저장한다.
기록·열람·종료·삭제 방법은 [관측 이미지 진단 안내](pose_observation_images.md)를 참고한다.
기존 좌표 전용 JSONL도 계속 열 수 있지만 과거 이미지는 복구할 수 없다.

## 현재 실행 동작 (UI 복원 후)

사용자 요청으로 게임 화면은 기존의 정면 스켈레톤 한 개씩 표시하는 방식으로 복원했다.
측면 보기용 코드는 진단 도구로만 남아 있다. 3D 좌표, 실제 시간 처리, 캘리브레이션,
최저점 분할 및 동일 가중치 비교는 유지한다.

실시간 자세 피드백과 REP 완료 후 정상/오류 판정, 정상 횟수 집계를 다시 활성화했다.
`assessment_supported`와 `reference_support`는 진단 기록으로만 사용하며 게임 판정을
보류시키지 않는다. 아래의 “판정 보류” 내용은 이전 진단 단계의 동작과 재생 결과다.
스켈레톤이 자연스럽게 보인다는 관찰과 실제 분류 정확도 검증은 구분한다.

실시간 게임은 `image_landmarks`를 화면에 표시하고, MediaPipe의
`world_landmarks`를 CommonSkeleton3DBridge로 변환해 판정한다.
TemporalLiftingNet의 2D→3D 추론은 이 경로에서 실행하지 않는다.
MediaPipe의 visibility는 영상에서 관절이 보일 가능성이며, Z좌표 정확도의
보증값이 아니다. [공식 출력 설명](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#output)

## 기록 방법

프로젝트 루트에서 새 파일명을 지정한다. 기존 파일은 덮어쓰지 않는다.

```powershell
.venv/Scripts/python.exe scripts/run_game.py --pose-trace output/diagnostics/squat_pose_02.jsonl
```

카운트다운 후 잠시 서 있다가 스쿼트를 수행하고 종료한다. 평소처럼 발을
바닥에 붙인 동작과 뒤꿈치를 의도적으로 든 동작을 별도 파일로 기록하면 비교할 수 있다.
기록은 세션이 활성화되고 프레이밍 검사를 통과한 프레임만 포함한다.
동영상 파일 대신 기록 대상 관측의 오버레이 없는 PNG와 좌표를 디스크에 기록한다.

- `image_landmarks`: MediaPipe 원본 2D 정규화 좌표와 visibility.
- `world_landmarks`: MediaPipe 원본 3D 추정치와 visibility.
- `common3d`, `frozen3d`: 필터/결측 보완 이후 18관절 좌표와 보완 여부.
- `aligned_frame`: 캘리브레이션 이후 실제 판정에 들어가는 좌표.
- `completed_rep`: 최종 클래스, DTW 거리, 실제 판정기(`classifier_source`),
  DL 확률(사용한 경우), 각 단계의 구간(`phase_bounds`), 동작 전체의 추정각 요약.
- 버전 2의 `sample_index`와 REP `frame_range`는 30Hz 보간 시퀀스 인덱스다.
  JSONL 행 번호(실제 카메라 관측 번호)와 다르다.
- `assessment_supported=false`이면 원시 후보 클래스(`predicted_class`)를 화면에
  확정 오류로 표시하지 않고 **판정 보류**로 표시한다. 성공/실패 집계에서도 제외한다.

MediaPipe 발 인덱스는 왼쪽/오른쪽 순으로 발목 27/28, 뒤꿈치 29/30,
발끝 31/32이다. Common Skeleton의 관절 순서는 `COMMON_JOINT_NAMES`를 따른다.

## 수정 및 해석상의 한계

발끝/뒤꿈치를 골반 기준 좌표에 고정하면, 스쿼트 중 추적되는 발목과 떨어져
발목 각도가 왜곡된다. 이제 발목에 대한 상대 위치를 필터링하고 결측 시 유지한다.
보완 좌표는 관측치가 아니며 `frozen3d`에 계속 표시된다. 실제로 관측된 발의
회전/뒤꿈치 들림은 허용한다. 바닥에 강제로 붙이는 보정은 적용하지 않는다.

최종 분류에 준비 구간이 빠져 DL이 항상 DTW로 대체되던 문제도 수정했다.
DL 체크포인트가 없거나 실제 단계가 너무 짧으면 기존대로 DTW를 사용한다.

원본 2D가 안정적인데 원본 3D 발 방향만 변하면 3D 추정의 문제를 의심할 수 있다.
원본 3D와 변환 후 좌표가 다르게 왜곡되면 필터/보완 과정을 확인한다.
좌표가 타당한데 오류로 분류되면 레퍼런스/분류기의 적합성을 검토한다.
각도 기반 분류 결과는 바닥 접촉의 직접 측정이 아니다. 실제 접촉 여부와
MediaPipe 3D의 정확성은 좌표 기록만으로 확정할 수 없으므로 동작 관찰과 함께 확인한다.

## 2026-09-20 기록 분석 및 추가 수정

`squat_pose_01.jsonl`은 310관측/약 30.9초, 실제 약 10.2fps였다. 이전 코드는
30fps 고정 필터/단계 임계값을 적용했다. 첫 캘리브레이션의 마지막 관측에서는
이미 무릎이 약 125~130도로 굽어 있었다. 4번째 REP의 최저점은 0프레임으로
기록되어 별도 최저점 비교에서 빠졌다.

- 필터에 실제 시각을 전달하고, 수신한 두 관측 사이만 30Hz로 보간한다.
  저속 입력에 없던 정보를 복원하는 것은 아니다. 0.5초 이상 추적이 끊기면
  중단된 REP를 버리고 서 있는 자세부터 재보정한다.
- 초기 캘리브레이션은 양쪽 무릎각이 150도 이상인 준비 자세에서만 진행한다.
  이 값은 준비 자세 판별용이며 스쿼트 정상/오류 기준이 아니다.
- 완료된 REP는 레퍼런스 생성과 같은 단계 분할기를 적용해 최저점을 포함한다.
- 클래스 순위를 정하는 DTW 거리는 모두 같은 가중치로 계산한다. 기존처럼
  클래스마다 다른 척도로 계산한 값을 직접 비교하지 않는다.
- 현재 4개씩의 레퍼런스로 클래스 내부 leave-one-out 최근접 거리의 최댓값을
  구한다. 후보까지의 거리가 이 경험적 범위를 넘으면 판정을 보류한다.
  이는 검증된 확률/정확도가 아니며, 통과해도 실제 자세의 정답을 보장하지 않는다.
  실제 라벨이 붙은 MediaPipe 녹화 데이터로 추가 검증해야 한다.
- 3D 화면은 정면(X/Y)과 측면(Z/Y)을 같은 고정 스케일로 표시한다. Z값 자체를
  임의로 축소하거나 관절을 정상 레퍼런스에 강제로 맞추지 않는다.
- 고관절오류를 무조건 “상체가 많이 기울었어요”로 해석하던 문구를 제거했다.
  완료 결과는 마지막 서 있는 프레임의 관절 오차 대신 해당 REP 전체의 추정각을 보여준다.

최종 코드를 같은 기록에 적용하면 5회 모두 인식되고, 5회 모두 레퍼런스 범위
검사를 통과하지 못해 판정 보류다. 이를 **정상 5회로 개선됐다고 해석하면 안 된다**.
원본 MediaPipe 3D와 필터 후 최저점 무릎각 차이는 약 0~2도였지만, 실제 사람의
각도를 측정한 영상이 없어서 원본 3D 추정의 정확성은 확인할 수 없다.

재현 명령:

```powershell
.venv/Scripts/python.exe scripts/inspect_pose_trace.py output/diagnostics/squat_pose_01.jsonl --out output/diagnostics/squat_pose_01_analysis.json
.venv/Scripts/python.exe scripts/replay_pose_trace.py output/diagnostics/squat_pose_01.jsonl --out output/diagnostics/squat_pose_01_replay.json
```

`--legacy-timing`은 30fps 고정 가정만 재현한다. 과거 버전 전체의 판정 코드를 재현하지는 않는다.

## 진단 전용 뼈 길이 제약 비교

게임 입력을 바꾸지 않고 `common3d`와 사용자별 길이 제약 결과를 비교할 수 있다.

```powershell
.venv/Scripts/python.exe scripts/compare_bone_length_constraint.py <trace.jsonl>
```

처음 발견되는 양쪽 무릎각 150도 이상의 8관측에서 좌우 허벅지·종아리와
발목–뒤꿈치–발끝 삼각형의 변 길이 중앙값을 구한다. 이후 Hip은 그대로 두고
Hip→Knee와 Knee→Ankle의 관측 방향을 보존하면서 기준 길이를 적용한다. 발은
세 변의 기준 길이가 유지되는 삼각형으로 재구성한다. 결과 영상과 `report.json`은
로그 폴더의 새 `bone_length_constraint` 폴더에 저장하며 기존 결과를 덮어쓰지 않는다.

이 결과는 원본 3D 정확도가 향상됐다는 증거가 아니다. 바닥면이나 발 접촉을 강제하지
않고, 오차가 있는 관절 방향도 그대로 보존한다. 현재 게임의 Phase·REP·DTW·CNN에는
연결되어 있지 않다. 좌표 이동량과 각도 보존 여부를 먼저 검토한 뒤에만 별도 회귀
실험을 해야 한다.

명시적인 시험 실행에서만 같은 보정을 실제 판정 입력에 적용할 수 있다. 기본 실행에는
영향이 없다.

```powershell
.venv/Scripts/python.exe scripts/run_game.py --bone-length-constraint --pose-trace <새 trace.jsonl>
```

이 모드의 로그는 필터 후 원본을 `common3d`, 실제 세션 입력을 `session_input3d`에
분리해 저장한다. 초기 기준이 확정되기 전에는 `session_input3d`가 `null`이며 Phase나
REP 세션에 좌표를 넣지 않는다. `bone_length_constraint`에는 활성화·보정 완료 여부와
보정에 사용한 관측 ID를 기록한다.
