# AI Trainer — 스쿼트 자세 교정 프로젝트
> 모든 사용자는 git 사용경험이 없는 사용자들이므로, git 작업 실행 전에 잘 알려주고 능동적으로 처리.
> 이 문서는 `제안서.md`의 계획 내용을 Claude Code용 프로젝트 문서로 정리한 것입니다.

## 1. 프로젝트 목적

웹캠을 통해 사용자의 스쿼트 자세를 실시간 분석하고, 올바른 자세와 비교하여 자세 교정 피드백을 제공한다.

## 2. 주요 기능

- 실시간 인체 스켈레톤 추출
- 스쿼트 동작 자동 검출
- 정상 자세와 사용자 자세 비교
- 자세 점수 및 오류 피드백 제공

## 3. 사용 기술

- **데이터**: AI Hub 스쿼트 자세 데이터
- **Pose Estimation**: MediaPipe Pose / YOLO Pose
- **개발**: Python, OpenCV
- **GUI**: PyQt

## 4. 자세 분석 방법

관절 좌표를 추출하여 무릎·고관절 각도, 상체 기울기, 좌우 균형 등을 계산하고, 정상 스쿼트 레퍼런스와 비교하여 자세의 정확도와 오류를 평가한다.

### 스쿼트 시퀀스

**준비 자세**
- 발 넓이: 어깨 너비, 골반 너비
- 발 끝: 10~30도 정도 자연스럽게 벌림
- 발은 전체적으로 지면과 맞닿을 것

**내려가는 동작**
- 엉덩이가 지면과 수평을 이루는 각과 같거나 더 낮아야 함
- 고관절 사용: 엉덩이를 뒤로 빼며 고관절 접기
- 무릎: 발 끝이 향하는 방향
- 발은 전체적으로 지면과 맞닿을 것

**올라오는 자세**
- 발은 전체적으로 지면과 맞닿을 것
- 준비 자세와 동일한 자세로 돌아갈 것

## 5. 개발 순서

1. AI Hub 스쿼트 데이터 분석 — 레퍼런스 스켈레톤 DB 구성
2. 웹캠 영상에서 포즈 스켈레톤 구현
3. 스쿼트 동작 검출
4. 레퍼런스와 비교
5. 자세 평가 알고리즘 구성
6. GUI 구성
7. 실험 및 성능 평가

- 이용자별 스켈레톤 좌표값 캘리브레이션 기능 추가 (고정값 부여에 따른 정확도 저하 예방)

### 프로그램 동작 순서

```
카메라 영상 입력 > Pose detecting > 3D skeleton 생성 > AI hub reference와 비교 > 자세 유사도 피드백
```

### AI Hub 데이터 사용 방안

```
스쿼트 데이터 추출 > 정상 데이터 지정 > 3D keypoint 지정 > reference skeleton DB 생성
```

### 자세 비교 방안

```
AI Hub 3D Keypoint > Common Skeleton Mapping > Hip-centered Translation >
Body-scale Normalization > Orientation Alignment > Biomechanical Feature Extraction >
Squat Segmentation > DTW > Similarity 계산 > Feedback
```
