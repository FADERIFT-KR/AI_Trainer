# 카메라 ray 기반 2D→3D lifting 후보

## 적용 범위

정면 단안 영상의 2D 관절 픽셀을 `K^-1 [u, v, 1]`의 단위 ray로 바꾸고, 9프레임
Temporal Lifting 모델에 입력하는 후보를 추가했다. 이는 Ray3D의 카메라-인지 입력 원칙을
따른다. ray는 방향만 나타내므로 한 장의 RGB 영상에서 절대 깊이를 직접 측정한다는 뜻은 아니다.

- MediaPipe 추론 전: ChArUco 보정으로 raw frame을 undistort한다.
- MediaPipe 추론 후: selfie mirror가 켜져 있으면 pixel x를 원래 카메라 좌표로 되돌린다.
- ray 변환: undistort에 실제 사용된 effective K를 사용한다. 원본 K를 재사용하지 않는다.
- 모델 출력: Hip-relative camera-coordinate 3D. 세션의 body-axis/scale 보정은 별도로 적용한다.

`SquatPipelineWorker`는 ray를 **판정 입력으로 바꾸지 않는다.** 보정된 카메라로 녹화한
세션의 JSONL에는 `effective_camera_matrix`, `camera_rays`, mirror 규약을 저장하여 추후
실제 웹캠 2D–3D 검증 데이터를 만들 수 있게 했다.

## AIHub offline ablation

AIHub 배포본에는 camera1의 K/R/t 파일이 없었다. 따라서 훈련 actor의 2D–3D 대응점으로
projective camera를 추정하고, validation actor에는 그 값을 고정해 적용했다. 이는 데이터셋
내부 비교를 위한 추정일 뿐, 사용자 웹캠 보정값은 아니다.

| 지표 (actor-disjoint, 동일 T=9 center frame) | 기존 2D pixel | ray 후보 |
| --- | ---: | ---: |
| MPJPE | 50.505 mm | 46.951 mm |
| dataset global z-MAE | 25.501 mm | 19.985 mm |
| ray camera-depth z-MAE | 해당 없음 | 31.535 mm |

훈련 actor로만 맞춘 투영모델의 held-out actor 재투영 오차는 median 6.06 px, p95 18.34 px였다.
따라서 이 *offline* 비교에서는 ray 후보를 선택했다. 자세한 기계 판정은
`output/ray_lifting_camera1/validation_comparison.json`과 `selection.json`에 남긴다.

## 안전한 배포 조건

현재 라이브 판정은 계속 MediaPipe `world_landmarks`를 사용한다. 후보를 바꾸려면 아래가
모두 필요하다.

1. 실제 사용 웹캠에서 `python scripts/calibrate_camera.py --camera <index>`로 ChArUco K/D를 만든다.
2. 같은 카메라·해상도·mirror 규약의 2D 키포인트와 깊이 카메라 또는 다중카메라 3D 정답을 수집한다.
3. 인물 분리 holdout에서 기존 라이브 소스보다 MPJPE, z-MAE, 스쿼트 최저점의 무릎/골반 z 오차를
   모두 개선하는지 확인한다.
4. T=9 중심 프레임 출력의 4프레임 지연을 session frame, DTW, replay timeline에 명시적으로
   보존한 뒤 feature flag로 전환한다.

이 조건을 충족하기 전에는 ray 후보를 실시간 DTW/오류 판정에 자동 사용하지 않는다.

## 재생성 명령

```powershell
python scripts/build_ray_lifting_dataset.py
python scripts/train_ray_lifting.py --epochs 30
python scripts/evaluate_ray_lifting.py
```

관련 연구: [VideoPose3D](https://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html), [Ray3D](https://openaccess.thecvf.com/content/CVPR2022/html/Zhan_Ray3D_Ray-Based_3D_Human_Pose_Estimation_for_Monocular_Absolute_3D_CVPR_2022_paper.html), [MotionBERT](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html).
