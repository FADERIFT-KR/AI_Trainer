# Runtime models

`pose_landmarker_lite.task`는 저장소에 포함하지 않습니다. 다음 명령으로 Google이 제공하는 MediaPipe Pose Landmarker lite model bundle을 내려받습니다.

```powershell
python scripts/download_pose_model.py
```

다운로드된 모델은 로컬 카메라 프레임에서 33개 2D landmark와 hip-centered 3D world landmark를 추론할 때 사용됩니다.
