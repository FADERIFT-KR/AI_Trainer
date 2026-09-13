# 실시간 스켈레톤과 레퍼런스 동작 매칭: 딥리서치 보고서

작성일: 2026-09-07  
대상: AI Trainer 에어스쿼트 판정 파이프라인

## 핵심 결론

현재 시스템을 DTW가 아닌 하나의 알고리즘으로 전면 교체하는 것은 권하지 않습니다. 가장 현실적인 개선안은 **현재 FSM과 phase-aware constrained DTW를 유지하면서 Soft-DTW divergence를 첫 비교 후보로 추가하고, 정상 여부를 확률 모델로 한 번 더 검증하는 하이브리드 구조**입니다.

추천 순서는 다음과 같습니다.

1. **즉시 실험:** constrained DTW 대 Soft-DTW divergence
2. **소규모 데이터 보강:** PCA/소형 Autoencoder + GMM 정상성 점수
3. **phase 확률화:** HMM 또는 지속시간을 명시하는 HSMM
4. **데이터가 충분해진 뒤:** supervised-contrastive ST-GCN 또는 EGCN++ 계열
5. **항상 유지:** 입력 신뢰도 검사, `판정 불가` 거부, 관절·phase별 설명 피드백

Soft-DTW는 DTW처럼 길이가 다른 시계열과 수행 속도 차이를 정렬하면서, 가능한 경로를 부드럽게 합산하고 미분 가능한 손실로 사용할 수 있습니다. 다만 계산량은 길이에 대해 제곱으로 증가합니다. 실제 거리 비교에는 자기 유사도 편향을 보정한 Soft-DTW divergence가 더 적합한 후보입니다. [Soft-DTW 원 논문](https://proceedings.mlr.press/v70/cuturi17a.html), [Soft-DTW divergence](https://proceedings.mlr.press/v130/blondel21a.html)

## 현재 프로젝트에 대한 진단

현재 코드는 이미 단순 DTW보다 발전된 구조입니다.

- 준비·하강·최저점·상승·완료로 나눈 phase-aware 비교
- phase별 12/24/12/24/12 프레임 리샘플링
- 정렬 경로 범위 25% 제한
- 사용자/레퍼런스 phase 길이비 3.0 초과 시 거부
- 정상 임계값을 먼저 적용하고, 정상에서 벗어난 경우에만 오류 class 선택
- 오류 레퍼런스 전체와 멀거나 1·2위 오류가 비슷하면 불안정/판정 불가 처리

로컬 [hybrid 평가 결과](../output/dtw_eval/offline_eval_report_hybrid.json)에서는 정규화된 DTW+DDTW의 `α=0.82`가 선택됐고 calibration balanced accuracy는 82.26%였습니다. 하지만 별도 validation 66개의 4-class accuracy는 59.09%이며, confusion matrix를 정상/오류 이진 문제로 합치면 48/66, 약 72.7%입니다. 즉 **calibration에서 고른 threshold의 점수와 실제 다중 오류 분류 성능을 같은 정확도로 해석하면 안 됩니다.**

이 차이는 알고리즘만의 문제가 아닙니다. 단안 카메라의 2D keypoint 오류가 3D lifting으로 전파되고, 정상 수행의 개인차와 오류 class 내부의 다양성이 큰 상황에서 가장 가까운 레퍼런스 하나로 class를 정해야 하기 때문입니다. 3D lifting 강건성 연구도 occlusion·motion blur·2D jitter가 downstream 3D 자세를 손상시키며, confidence-aware 처리와 2D pose-domain noise augmentation이 도움이 된다고 보고합니다. [Hoang et al., CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/papers/Hoang_Improving_the_Robustness_of_3D_Human_Pose_Estimation_A_Benchmark_CVPRW_2024_paper.pdf)

## 후보 기법 비교

| 기법 | 시간 차이 처리 | 적은 데이터 | 실시간성 | 오류 설명 | 이 프로젝트 판단 |
|---|---:|---:|---:|---:|---|
| Constrained DTW/DDTW | 높음 | 높음 | 높음 | 높음 | 유지할 기준선 |
| Soft-DTW divergence | 높음 | 높음 | 중간~높음 | 높음 | 1순위 추가 후보 |
| GMM 정상성 모델 | 간접적 | 중간~높음 | 높음 | 중간 | 정상/오류 1차 게이트에 적합 |
| HMM/HSMM | 높음 | 중간 | 높음 | 높음 | FSM 보완 또는 대체 후보 |
| Global Alignment Kernel + SVM | 높음 | 높음 | 중간 | 낮음~중간 | 소규모 분류 실험 후보 |
| ST-GCN/EGCN++ | 학습함 | 낮음~중간 | 추론은 높음 | 중간 | 데이터 확장 후 후보 |
| Transformer | 학습함 | 낮음 | 중간 | 낮음~중간 | 현재는 우선순위 낮음 |
| 규칙+관절 통계 | 제한적 | 높음 | 매우 높음 | 매우 높음 | 구체적 피드백 보조기 |

### 1. Soft-DTW divergence

일반 DTW는 비용이 가장 작은 경로 하나를 선택하므로, 잡음이 있는 관절을 반복 매칭하며 비현실적인 경로를 만들 수 있습니다. Soft-DTW는 여러 경로를 soft-min으로 합산해 경로 선택을 완화합니다. 또한 학습 손실로 쓸 수 있어, 나중에 스켈레톤 embedding 모델을 학습할 때도 같은 정렬 개념을 유지할 수 있습니다. [Cuturi & Blondel, ICML 2017](https://proceedings.mlr.press/v70/cuturi17a.html)

현재 코드에 적용할 때는 FSM과 phase 분할을 바꾸지 않고 각 phase의 `_dtw_dp`만 Soft-DTW divergence로 교체하면 됩니다. `gamma`는 고정 이론값이 없으므로 표준화된 local cost에서 `0.01, 0.03, 0.1, 0.3, 1.0`을 calibration actor로만 탐색해야 합니다. DTW와 같은 warping window 및 phase-length rejection을 유지해야 공정하게 비교할 수 있습니다.

### 2. GMM 또는 저차원 정상 분포 모델

DTW는 “이 레퍼런스와 얼마나 먼가”를 묻지만 GMM은 “정상 동작 분포에서 이 수행이 나올 가능성이 얼마나 되는가”를 묻습니다. UI-PRMD의 10개 재활 운동을 대상으로 한 연구에서 Autoencoder 저차원 표현의 GMM log-likelihood는 정상과 오류의 separation degree가 between-subject 0.515, within-subject 0.603이었고, 같은 표현의 DTW는 각각 0.427, 0.574였습니다. 저자들은 확률 모델이 사람 움직임의 변이와 측정 불확실성을 더 잘 다룬 결과로 해석했습니다. [Liao et al., IEEE TNSRE 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7032994/)

이 프로젝트에서는 난이도별 정상 레퍼런스 중 하나와 가깝기만 하면 정상으로 보는 대신, 초·중·고급 정상 데이터를 하나의 mixture로 모델링할 수 있습니다. 다만 표본이 적으므로 full covariance GMM은 피하고 diagonal covariance, 강한 regularization, actor-disjoint 검증이 필요합니다.

### 3. HMM/HSMM

HMM은 `Standing → Descending → Bottom → Ascending → Standing` 순서를 숨은 상태로 보고, 관절 특징이 각 상태에서 발생할 확률과 상태 전이를 함께 학습합니다. HSMM은 여기에 각 phase의 지속시간 분포를 명시하므로, 사용자마다 다른 속도는 허용하면서 지나치게 짧거나 긴 phase를 확률적으로 낮출 수 있습니다. HMM을 사용한 실시간 재활 플랫폼 연구는 관절/관절군별 모델로 sequence mismatch와 수행 품질을 평가했습니다. [Vourvopoulos et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5982640/)

현재 FSM의 설명 가능성과 안전한 카운팅은 장점이므로 즉시 대체하기보다는 다음처럼 보조 점수로 쓰는 편이 좋습니다.

`최종 정상성 = 정렬 거리 점수 + HMM/HSMM log-likelihood + 입력 신뢰도`

### 4. Global Alignment Kernel과 공간-시간 정렬

Global Alignment Kernel은 최소 경로 하나가 아니라 가능한 정렬 경로를 종합한 similarity를 만들어 SVM 같은 분류기에 사용할 수 있습니다. 레퍼런스가 적은 환경에서 실험할 가치가 있지만, 어떤 frame과 관절이 오류였는지 설명하기는 DTW보다 어렵습니다. [Cuturi et al., Global Alignment Kernel](https://arxiv.org/abs/cs/0610033)

관절 정의나 좌표 공간 자체가 서로 어긋나는 경우에는 optimal transport로 공간 대응을 찾고 Soft-DTW로 시간을 맞추는 Spatio-Temporal Alignment도 있습니다. 그러나 해당 논문의 실험은 스쿼트가 아니고, 현재 시스템은 공통 18관절 mapping이 이미 고정돼 있으므로 당장 우선순위는 낮습니다. [Janati et al., AISTATS 2020](https://proceedings.mlr.press/v108/janati20a.html)

### 5. ST-GCN, contrastive learning, EGCN++

GCN 계열은 스켈레톤의 관절을 node로, 뼈 및 시간 연결을 edge로 보아 관절 간 협응과 시간 변화를 함께 학습합니다. 2s-AGCN은 joint 위치와 bone 방향을 두 stream으로 처리하는 것이 상호보완적임을 보였습니다. 다만 이 결과는 대규모 action-recognition 데이터에서 나온 것이므로 바로 자세 품질평가 성능을 뜻하지는 않습니다. [2s-AGCN, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html)

재활 품질평가에 직접 적용한 연구는 exercise별 표본이 적다는 문제를 명시하고, 여러 운동의 hard/soft negative를 활용하는 supervised contrastive ST-GCN을 제안했습니다. UI-PRMD, IRDS, KIMORE에서 평가됐다는 점에서 일반 action-recognition 논문보다 현재 과제와 가깝습니다. [Karlov et al., 2024](https://arxiv.org/abs/2403.02772)

위치와 orientation을 함께 융합한 EGCN++는 cross-subject 조건에서 UI-PRMD 89.95%, KIMORE 83.56%, EHE 86.38%의 accuracy를 보고했습니다. 그러나 exercise 종류, label 정의, sensor, split이 현재 AI Hub 에어스쿼트와 다르므로 이 수치를 현재 59.09%와 직접 비교할 수는 없습니다. [EGCN++, IEEE TPAMI 2024](https://ieeexplore.ieee.org/document/10475587/)

따라서 GCN은 **actor 수와 오류 수행 다양성을 확대한 뒤** 도입하는 것이 맞습니다. 현재 데이터 규모에서는 큰 모델보다 작은 ST-GCN encoder에 supervised contrastive loss를 적용하고, 최종 layer를 거리 기반 prototype classifier로 두는 구성이 더 안전합니다.

### 6. Joint Relation Graph와 스쿼트 전용 규칙

Action Quality Assessment에서는 단순 action 종류가 아니라 인접 관절의 협응 차이를 봐야 합니다. Joint Relation Graph 연구는 spatial/temporal relation graph로 이웃 관절의 공통 움직임과 차이를 학습하고, 관계 가중치를 통해 평가 근거를 해석할 수 있음을 보였습니다. 다만 검증 대상은 올림픽 종목 영상이라 스쿼트에 대한 직접 증거는 아닙니다. [Pan et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Pan_Action_Assessment_by_Joint_Relation_Graphs_ICCV_2019_paper.html)

스쿼트에 한정하면 2026년 연구는 MediaPipe로 hip/knee angle을 측정하고, hip-knee coordination 회귀와 깊이 tolerance를 결합한 경량 모델로 자체 실험에서 90% accuracy를 보고했습니다. 대규모 신경망 없이 관절 협응을 보는 접근은 현재의 고관절·엉덩이 하방 피드백과 잘 맞습니다. 다만 독립 재현 전에는 90%를 일반 성능으로 간주하면 안 됩니다. [Yao et al., Computers 2026](https://www.mdpi.com/2073-431X/15/5/293)

## 데이터셋 연구가 주는 경고

UI-PRMD는 deep squat을 포함하며, 정상 squat을 “뒤꿈치가 바닥에 있고 무릎이 발 위에 정렬되며 상체가 수직면에 유지되는 수행”으로 정의합니다. 그러나 참가자는 건강한 성인 10명뿐이고 잘못된 동작도 이들이 모사했습니다. 원 저자도 실제 환자가 오류 동작을 수행하지 않았다는 한계를 명시합니다. [UI-PRMD 원 데이터 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC5773117/)

KIMORE는 저요통 재활 운동 5개, 78명—건강 44명과 운동기능장애 34명—의 RGB·depth·skeleton 및 임상 점수를 제공합니다. 실제 환자와 임상 점수가 있다는 장점은 있지만, 현재의 에어스쿼트 오류 3종과 같은 label 체계는 아닙니다. [KIMORE 원 논문](https://pubmed.ncbi.nlm.nih.gov/31217121/)

IRDS는 29명 중 15명이 환자이고 14명이 건강 대조군이며, 9개 재활 gesture의 25개 3D 관절과 correctness label을 제공합니다. 이들 데이터셋은 논문 성능을 복사하기보다 **실제 사용자, 실제 오류, actor-disjoint 분할이 중요하다**는 근거로 활용해야 합니다. [IRDS 원 논문](https://kclpure.kcl.ac.uk/portal/en/publications/intellirehabds-irdsa-dataset-of-physical-rehabilitation-movements/)

## 권장 최종 구조

```text
카메라 프레임
  → 2D pose + 관절 confidence
  → confidence-aware EMA / 짧은 결손 보간
  → 3D lifting + 2D 지면 상대 특징 병행
  → FSM으로 완전한 1회와 5개 phase 확정
  → 정상 게이트
       ├─ constrained DTW / Soft-DTW divergence
       └─ GMM 또는 HMM/HSMM likelihood
  → 정상 threshold 통과: 정상 + 횟수 카운트
  → 정상 threshold 실패: 오류 reference 분류
       ├─ 오류 거리 통과 + 1·2위 margin 충분: 오류 확정
       └─ 아니면: 판정 불가 / 동작 인식 불안정
  → phase·관절별 비용 + 명시적 특징으로 피드백
```

이 구조에서 발뒤꿈치·엉덩이 하방·고관절 오류는 최종 class를 억지로 고르는 기준이 아니라, 정상 게이트를 벗어난 뒤 오류 근거를 설명하는 계층이 됩니다. 입력 confidence가 낮으면 운동 오류로 분류하지 않고 인식 불안정으로 돌려야 합니다. 불확실한 상황에서 분류를 거부하는 것은 안전이 중요한 분류 문제에서 오판 비용을 낮추는 정식 설계 선택입니다. [Franc & Prusa, ICML 2019](https://proceedings.mlr.press/v97/franc19a.html)

## 검증 계획

동일한 actor-disjoint split에서 아래 실험을 순서대로 실행하는 것이 좋습니다.

1. DTW, DDTW, 현재 hybrid, Soft-DTW divergence를 같은 feature·phase·window로 비교
2. PCA/소형 AE + diagonal GMM 정상성 점수 추가
3. HMM/HSMM likelihood를 정상 게이트의 보조 점수로 추가
4. confidence weighting, frame dropout, jitter augmentation ablation
5. 데이터 확장 후 supervised-contrastive ST-GCN 비교

최소 지표는 다음과 같습니다.

- 정상/오류 balanced accuracy와 정상 recall
- 오류 3종 macro-F1 및 confusion matrix
- `판정 불가`를 포함한 coverage–selective risk
- actor 단위 bootstrap 신뢰구간
- phase boundary error와 1회 카운팅 정확도
- 640×480 실제 장치에서 end-to-end FPS와 종료 후 판정 지연
- 관절 jitter·occlusion·confidence 구간별 성능

threshold, Soft-DTW `gamma`, hybrid `α`, 오류별 margin은 calibration actor에서만 선택하고 test actor에는 고정해야 합니다. 같은 validation 집합으로 값을 고르고 그 집합의 성능을 최종 성능으로 보고하면 낙관적 편향이 생깁니다.

## 최종 권고

가장 비용 대비 효과가 큰 다음 작업은 **Soft-DTW divergence를 현재 phase-aware 인터페이스에 추가해 actor-disjoint 재평가하는 것**입니다. 그 다음 **정상 레퍼런스를 GMM 분포로 모델링해 정상/오류 게이트를 강화**하십시오. ST-GCN/Transformer는 데이터가 충분해진 뒤의 단계이며, 지금 바로 도입하면 모델 용량보다 사용자·오류 다양성 부족이 먼저 문제가 될 가능성이 큽니다.

공개 문헌에서 현재 AI Hub 관절셋, 오류 3종, 단안 lifting, 동일 FSM 조건으로 모든 방법을 직접 비교한 연구는 확인되지 않았습니다. 따라서 위 순위는 논문의 타 데이터셋 성능을 그대로 이전한 결론이 아니라, 현재 코드 구조·데이터 규모·실시간 피드백 요구를 함께 고려한 연구 기반 제안입니다.
