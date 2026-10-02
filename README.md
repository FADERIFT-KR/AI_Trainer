# Squat Error Data Augmentation

AI Hub 에어스쿼트의 정상·실제 오류 3D 스켈레톤만 사용해 조건부 ML 생성기를
학습하고, 정상 스쿼트에서 세 종류의 합성 오류 스켈레톤을 만드는 최소 브랜치다.
영상·GUI·실시간 판정 기능은 포함하지 않는다.

## 포함 범위

1. AI Hub zip 또는 추출 디렉터리에서 2D/3D 라벨 읽기
2. 26관절을 공통 18관절로 매핑하고 노이즈 보정
3. 사용자 단위 학습/검증 분리
4. 64프레임 정규화와 하강·최저점·상승 구간 추출
5. 실제 오류 데이터로 오류 판별기와 생체역학 특징 분포 학습
6. 정상 동작에서 조건부 오류 잔차 생성
7. 뼈 길이·발 접지·시간 연속성과 오류 의미 검증
8. 검증을 통과한 3D 및 사선 2D 스켈레톤을 NPZ로 저장
9. 예측 3D와 독립적인 정답 3D의 관절·Z축 오차 계산

오류 라벨은 `발뒤꿈치오류`, `엉덩이하방오류`, `고관절오류`다.

## 학습

```powershell
python -m ai_trainer.augmentation.train_cli `
  --dataset D:\AIHub\TL.zip `
  --dataset D:\AIHub\VL.zip `
  --output models\error_augmenter.pt
```

학습에 사용하지 않은 actor가 검증 세트가 된다. 체크포인트 옆 JSON에는 실제
검증 정확도, 생성 표본 통과율, 학습된 오류 특징 분포가 기록된다.

## 생성

```powershell
python -m ai_trainer.augmentation.generate_cli `
  --dataset D:\AIHub\TL.zip `
  --dataset D:\AIHub\VL.zip `
  --split output\error_augmentation_actor_split.json `
  --checkpoint models\error_augmenter.pt `
  --output output\generated_errors
```

생성 데이터는 `NPZ + manifest.jsonl` 형식이다. manifest에는 원본 actor/반복,
오류 라벨·강도, 합성 여부 및 검증 결과가 저장된다. 기존 결과 디렉터리는 자동으로
덮어쓰지 않는다.

## 중요한 제한

- 이 브랜치는 AI Hub 내부 증강을 실행할 수 있지만 외부 카메라·체형 일반화를
  입증하지 않는다.
- 합성 데이터는 학습 보조용이며 실제 오류 검증 세트를 대체하지 않는다.
- 2D 투영은 신체 기준 직교 투영이다. 실제 카메라 픽셀을 재현하려면 별도의 카메라
  내부·외부 파라미터가 필요하다.
- 생성기가 실제 오류 분포와 구조 검사를 모두 통과하지 못하면 표본을 저장하지 않는다.
- 고관절 오류 합성 검증의 후보 특징에 고관절각·상체 기울기와 함께 상체-정강이의
  전후 기울기 차이를 포함한다. 고정 각도 임계값 대신 학습 세트의 라벨별 분포를
  사용하지만, 이 조합이 실제 오류 판정 성능을 개선하는지는 별도 비교가 필요하다.
  이 특징은 정답 3D의 신체 기준 좌표에서 정의되므로 2D 영상 각도에 그대로 적용하지 않는다.
  특징 구성이 바뀌었으므로 이전 생성기 체크포인트는 재학습해야 한다.
- `ai_trainer.augmentation.metric_validation.compare_sequences`는 동일 프레임·단위(m)·
  좌표축의 18관절 추정/정답 배열을 비교한다. 영상 파일은 사용하지 않는다. 정답 3D가
  없는 외부 데이터에는 수치 정확도를 계산할 수 없다. PA-MPJPE는 매 프레임 회전·크기를
  맞추므로 Z축 정확도 근거로 쓰지 않는다.
