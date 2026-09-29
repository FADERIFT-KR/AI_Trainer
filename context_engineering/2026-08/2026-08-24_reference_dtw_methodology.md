정상 자세 비교/평가 방법론

AI Hub - https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71422 (크로스핏 동작 데이터)

AI Hub - 국가 AI 학습데이터 허브 (전문가가 라벨링한 3D/2D 관절 좌표 + 동작 구간 주석 제공)

AI Hub 크로스핏 데이터는 3d_points.csv(3D 관절), camera별 local_keypoints/*.csv(2D 관절), annotation.json(동작 구간·피험자 정보) 세 가지 라벨링 파일로 구성되며
원천 영상/이미지는 전혀 사용하지 않고 이 라벨링 데이터만으로 정상·오류 레퍼런스를 구축한다

또 데이터는 actor(피험자) × repetition(반복) × camera(8대) 조합으로 구성되며
이 중 정면 카메라 1대(camera1)를 후보로 정량 검증(Procrustes 기반)해 학습에 사용한다

동작 구조는 데이터 선별 - 정규화 - phase 분할 - 대표(medoid) 선정 - DTW 비교 - 채점/피드백 으로 구성된다
입력으로 AI Hub 라벨링 CSV/JSON과 실시간 웹캠 2D 관절 두 가지를 지원한다(같은 파이프라인 공유)

--------------레퍼런스/비교 파이프라인 구성--------------

2D→3D Temporal Lifting
- camera1(정면 후보)의 2D 좌표를 입력으로 3D 좌표를 추정
- 경량 Dilated Temporal Conv1d, T=9 시간창 사용
- AI Hub 3D CSV를 Ground Truth로 지도학습 (val MPJPE 48.7mm)

정규화 파이프라인 (카메라/사용자 무관하게 비교 가능하도록)
- Hip-centered Translation: 골반을 원점으로 이동
- Body-scale Normalization: 다리 길이로 나눠 신장·체형 차이 제거
- Orientation Alignment: 좌우 Hip 벡터·Pelvis→Shoulder 벡터로 body-centered 좌표계 정렬

Phase Segmentation
- pelvis 높이·속도 기반 규칙 기반 상태기계
- 준비 - 하강 - 최저점 - 상승 - 종료 5단계로 분할

Multi-Reference DB
- 정상 / 발뒤꿈치오류 / 엉덩이하방오류 / 고관절오류 4클래스
- 클래스 내부 pairwise DTW 거리 + k-medoids로 대표 시퀀스 4개 선정 (단일 평균 스켈레톤 대신 실존 시퀀스 사용)
- Ground Truth Reference(AI Hub 3D 그대로) / Operational Reference(2D→3D lifting을 거친 것) 2계층 관리

Weighted Phase-aware DTW
- 관절좌표·관절각·본(bone) 방향벡터·속도·궤적 등 11개 feature의 가중합으로 프레임 간 거리 정의
- phase별로 독립 DTW 계산 후 가중합(최저점 구간 등 가중치 상향)
- 여러 레퍼런스와 비교해 min/top-k distance로 클래스 판별, distance→점수 변환

Online DTW (실시간 세션)
- 미래 프레임 미참조(causal), T=9로 인한 고정 4프레임 지연만 존재
- phase 실시간 추정, rep 시작/종료 자동 검출
- rep 종료 시 Offline Weighted DTW로 최종 평가

--------------라이브러리별 역할--------------

PyTorch
- 2D→3D Temporal Lifting 모델 정의·학습·추론

NumPy / SciPy
- 정규화(회전행렬, 벡터 연산), 관절각 계산
- Procrustes 분석(정면 카메라 후보 검증), cdist(feature별 거리행렬)

fastdtw
- feature 가중 거리행렬을 입력받아 DTW 정렬 및 거리 계산

OpenCV
- 스켈레톤 렌더링(관절점·연결선), 카메라 프레임 처리, 화각 가이드 박스·말풍선 오버레이

PyQt5 (game_ui 통합)
- 좌(웹캠+내 스켈레톤)/우(정상 레퍼런스) 2분할 GUI
- QThread + Qt Signal로 GUI와 카메라·추론 스레드 분리
- 3-2-1 카운트다운으로 세션 시작 시점(캘리브레이션 시작점) 고정

--------------프로그램 실행 시 출력--------------

좌: 실시간 카메라 + 2D 관절 오버레이 + 화각 가이드박스
우: 정상 레퍼런스 스켈레톤 (현재 phase에 맞춰 재생)
현재 phase 및 rep 카운트 표시
클래스별 partial DTW distance / 실시간 정오 판정
오류 관절 옆 말풍선 설명 (예: 고관절오류 → 고관절 근처)
rep 종료 시 최종 판정 클래스 · 점수 · 주요 기여 feature
