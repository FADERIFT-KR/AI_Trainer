# AI Trainer

소재부품융합공학과 졸업프로젝트 5조의 실시간 스쿼트 자세 분석기입니다.

## 현재 구현 범위

- 웹캠 영상에서 MediaPipe Pose 관절선 표시
- 좌우 무릎 및 고관절 각도 실시간 계산
- 측정 결과 CSV 기록
- 카메라와 분리된 각도 계산 단위 테스트

## 실행 환경 준비

Python 3.11 설치를 권장합니다. 프로젝트 폴더에서 다음 명령을 실행합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 실행

```powershell
python -m src.main
```

- `R`: CSV 기록 시작 또는 종료
- `Q`: 프로그램 종료
- 기본 카메라가 아닌 경우: `python -m src.main --camera 1`

CSV 파일은 `recordings/`에 생성되며 Git에는 포함되지 않습니다.

## 테스트

```powershell
python -m unittest discover -s tests -v
```
