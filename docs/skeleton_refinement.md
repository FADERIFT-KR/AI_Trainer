# 3D 스켈레톤 Jitter/Spike 정제 근거와 검증

## 채택한 처리 순서

AI Hub `3d_points.csv`는 `AiHubZip.read_3d()`에서 기본적으로 다음 정제를 거친다. 원본 감사가 필요할 때만 `read_3d(..., refine=False)`를 사용한다.

실시간 `내 3D 스켈레톤`에는 오프라인 필터가 적용되지 않는다. MediaPipe 좌표를 공통 관절로 변환할 때 좌우 고관절과 무릎에 동일한 역할별 점프 검사(`game_ui/spike_guard.py`)를 먼저 적용하고, 그 다음 기존 1€ 필터를 적용한다. 가시성이 떨어진 뒤 재인식된 관절의 단발성 큰 이동은 이전 상대 위치에 머물게 한다. 여러 프레임 동안 계속 관측되며 뼈 길이도 타당한 움직임은 프레임당 다리 길이의 최대 8%(무릎) 또는 4.5%(고관절)씩 따라간다. 이 한계는 30fps 기준이며 실제 웹캠 녹화로 추가 조정이 필요하다.

1. **Spike 분리**: 7프레임 rolling median과 MAD(Hampel 계열 robust scale)로 관절별 3D 이상점을 찾고, 정상 이웃 프레임 사이를 보간한다.
2. **Jitter 저감**: 30 fps, 6 Hz cutoff, 4차 Butterworth를 forward/backward로 적용한다. 따라서 위상 지연은 0이고 스쿼트 최저점 시각이 밀리지 않는다.
3. **인체 구조 안정화**: Spike 제거 후 얻은 시퀀스별 median bone length로 주요 사지를 65% soft projection한다. 방향은 필터 결과를 유지하고 길이 변동만 줄인다.

구현은 [`ai_trainer/skeleton_filter.py`](../ai_trainer/skeleton_filter.py), 전체 데이터 감사는 `python scripts/audit_3d_skeleton_refinement.py`로 재현할 수 있다.

## 핵심 문헌과 선택 이유

- Butterworth, *On the Theory of Filter Amplifiers* (1930): 최대 평탄 저역통과의 기초. [원문 PDF](https://people.eecs.ku.edu/~demarest/212/Buttorworth%20Orig.%20Paper.pdf)
- Winter, Sidwall, Hobson, *Measurement and reduction of noise in kinematics of locomotion* (1974): 인체 marker trajectory에서 신호/노이즈 주파수를 분리하고 low-pass 처리해야 속도·가속도를 신뢰할 수 있음을 보였다. [DOI](https://doi.org/10.1016/0021-9290(74)90056-6)
- Hampel, *The Influence Curve and its Role in Robust Estimation* (1974): MAD 기반 이상점 판별의 robust statistics 토대. [DOI](https://doi.org/10.1080/01621459.1974.10482962)
- Zhang et al., *A statistical smoothness measure to eliminate outliers in motion trajectory tracking* (1998): tracking outlier는 일반 low-pass만으로 제거할 수 없으므로 smoothing 전에 별도로 처리해야 한다. [DOI](https://doi.org/10.1016/S0167-9457(97)00029-8)
- Savitzky & Golay, *Smoothing and Differentiation of Data by Simplified Least Squares Procedures* (1964): 국소 다항 least-squares smoothing의 고전. 짧아서 Butterworth `filtfilt`를 적용할 수 없는 시퀀스의 fallback과 phase 분석에 사용한다. [DOI](https://doi.org/10.1021/ac60214a047)
- Woltring, *A FORTRAN package for generalized, cross-validatory spline smoothing and differentiation* (1986): biomechanics에서 data-driven smoothing parameter를 선택하는 대표 방법. [DOI](https://doi.org/10.1016/0141-1195(86)90098-7)
- Casiez, Roussel, Vogel, *1€ Filter* (CHI 2012): 속도에 따라 cutoff를 높여 정지 jitter와 동작 lag를 절충한다. 오프라인 AI Hub 정제보다 실시간 MediaPipe 2D/3D 경로에 이미 적용돼 있다. [DOI](https://doi.org/10.1145/2207676.2208639)
- Kalman, *A New Approach to Linear Filtering and Prediction Problems* (1960): 상태공간 최적 필터의 고전. [DOI](https://doi.org/10.1115/1.3662552) 현재 데이터에는 관절별 process/measurement covariance가 없고 오프라인 zero-phase가 가능하므로 기본 정제기로는 채택하지 않았다.

## 전체 361개 시퀀스 감사 결과

[`configs/skeleton_refinement_report.json`](../configs/skeleton_refinement_report.json)에 모든 수치와 파라미터를 저장한다.

- 정규화 jerk RMS 평균: **90.86% 감소**
- bone-length CV 평균: **65.27% 감소**
- Spike 판정 비율 평균: **3.13%**
- 원본 대비 좌표 RMS 수정량: median bone length의 **3.16%**
- 스쿼트 최저점 중심 이동: 평균 **0.5 frame**
- 무릎 굽힘각 RMS 변화: 평균 **0.87°**

Jitter 감소만 보면 과평활 가능성이 있으므로, 최저점 frame과 무릎각 보존 지표를 함께 통과한 현재 파라미터를 사용한다. 정제 후 actor-disjoint reference DB를 다시 구축했으며, 검증 66개 시퀀스에서 DTW 4-class 정확도 65.2%, 정상/오류 정확도 87.9%였다. 이 수치는 실시간 웹캠 전체 시스템이 아니라 정제된 CSV 기반 DTW 검증 결과다.
