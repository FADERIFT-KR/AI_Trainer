# 같은 관측의 이미지와 좌표 진단

`--pose-trace`를 켰을 때만 로컬 JSONL과 관측 PNG를 기록합니다.
기존 게임 화면, REP/DTW/CNN/점수 알고리즘은 변경하지 않습니다.
단, PNG 인코딩과 파일 저장을 카메라 워커에서 동기 처리하므로 실제 처리 FPS가
낮아질 수 있고, 입력 관측 간격 변화는 보간 및 최종 판정에 간접 영향을 줄 수 있습니다.
진단 실행과 비진단 실행의 결과가 항상 같다고 보장하지 않습니다.

## 다음 촬영 실행 (PowerShell)

```powershell
Set-Location C:\AI_Trainer
$poseSessionDir = Join-Path 'C:\AI_Trainer\output\diagnostics' ('capture_' + [guid]::NewGuid().ToString('N'))
$poseTracePath = Join-Path $poseSessionDir 'trace.jsonl'
$poseSessionDir
.\.venv\Scripts\python.exe -B .\scripts\run_game.py --pose-trace "$poseTracePath"
```

워커가 새 폴더와 JSONL을 생성합니다. 기록 폴더 구조:

```text
C:\AI_Trainer\output\diagnostics\capture_<고유 ID>\
  trace.jsonl
  trace_images_<고유 ID>\
    observation_00000042.png
    observation_00000043.png
    ...
```

JSONL과 PNG 모두 배타적 새 파일 생성으로 기존 파일을 덮어쓰지 않습니다.
이미지 경로는 JSONL 폴더 기준 상대 경로이므로 폴더 전체를 함께 이동할 수 있습니다.
5회 완료 후 또는 게임 창을 닫거나 Esc로 비교 화면을 나가면 기록이 종료됩니다.
같은 프로세스의 다시 하기는 같은 JSONL 경로를 열려다 거부될 수 있으므로,
다음 기록은 게임을 종료하고 위 명령으로 새 경로를 만들어 시작하세요.

## 무엇을 기록하는가

- 기존과 같이 세션 활성화·프레이밍 통과·관절 검출 조건을 만족한 관측만 기록합니다.
- PNG는 관절, 안내 박스, 텍스트를 그리기 전 이미지입니다. 무손실 BGR PNG이며,
  RGB 채널 순서를 제외하면 MediaPipe에 입력한 이미지와 같습니다.
- 기본 좌우 반전 설정을 따릅니다. JSONL의 `observation_image.mirrored`와
  `orientation="mediapipe_input"`, `overlay=false`에 이를 명시합니다.
- `observation_id`: 워커에서 성공적으로 읽은 카메라 프레임마다 증가하는 0-based 번호.
  기록 제외 구간 때문에 번호가 건너뛸 수 있고, 새로운 워커에서는 다시 시작합니다.
  새 세션 폴더/JSONL과 ID의 조합으로 관측을 구분합니다.
- `timestamp`: 해당 카메라 읽기 직후 perf_counter 시각(초). 달력 시각이나
  카메라 하드웨어 노출 시각은 아닙니다.
- `sample_index`: 마지막으로 처리한 30Hz 보간 샘플 번호. observation_id와 다릅니다.
- JSON 행과 이미지 메타데이터에 동일 observation_id를 저장하고 PNG 파일명에도 넣습니다.
  이미지 SHA-256과 크기도 검증하여 잘못 연결되거나 변경된 파일은 표시하지 않습니다.
- Image/World와 PNG는 같은 관측입니다. aligned_frame은 마지막 보간 샘플이므로
  정확히 같은 관측 시각이라고 간주하지 않습니다. 누락된 샘플은 생성하지 않습니다.

## 보기

게임 종료 후 같은 터미널:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\view_pose_trace.py "$poseTracePath"
```

관측 행을 이동하면 원본 이미지 / Image 관절을 겹친 이미지 / World XY·ZY가 함께 바뀝니다.
2D 오버레이는 기존 draw_2d_pose를 사용하며 visibility 0.45 미만은 그리지 않습니다.
원본 PNG는 수정하지 않습니다. 이전·다음/좌우 화살표/행 번호 직접 입력을 사용할 수 있습니다.
행 번호는 JSONL 0-based 행 번호이며 observation_id나 sample_index가 아닙니다.
기존 좌표 전용 로그는 이미지 기록 없음으로 표시하며 3D와 수치 표는 그대로 동작합니다.

## 실패·보관·삭제

이미지를 먼저 저장하고 JSON 행을 기록합니다. 인코딩·파일 저장 실패는 기존 게임의
오류 표시 경로로 명확하게 알리고 현재 워커/세션을 중단합니다. 저장 실패를 무시하고
다른 좌표로 판정을 계속하지 않습니다. 이미 계산 중이던 마지막 결과는 화면/로그에
전달되지 않을 수 있으며 해당 실행은 중단된 진단으로 취급해야 합니다.
실패/강제 종료 시 연결되지 않은 PNG 또는 불완전한 마지막 JSON 행이 남을 수 있습니다.
오류를 해결한 후 새 세션 경로로 시작하세요. 실패 파일을 자동 삭제하지 않습니다.

PNG에는 사람의 모습과 주변 공간이 포함됩니다. 파일은 위 로컬 폴더에 저장되며,
진단 코드에서 외부로 전송하지 않습니다. 필요 없어지면 게임·뷰어를 닫고 탐색기에서
정확한 `capture_<고유 ID>` 폴더를 확인한 뒤 그 폴더를 삭제하면 JSONL과 PNG가 함께 삭제됩니다.
`output\diagnostics` 전체나 다른 기존 로그를 삭제하지 마세요.

진단 없이 실행:

```powershell
Remove-Item Env:AI_TRAINER_POSE_TRACE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -B .\scripts\run_game.py
```

이 경우 이번 이미지 기록 기능은 호출되지 않습니다. 위 Remove-Item은 환경변수만 지우며
저장된 기록 파일을 지우지 않습니다.
