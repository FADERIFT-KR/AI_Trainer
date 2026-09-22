"""MediaPipe 포즈 추정 — 공정 2.

BGR 프레임 -> 33관절 2D(image_landmarks) + 3D(world_landmarks).
mediapipe_pose가 모델을 직접 돌리고, frame_processor는 캡처~추정~그리기를
한 번에 묶은 단독 뷰어용 래퍼다. 좌표를 만드는 곳은 여기가 유일하다."""
