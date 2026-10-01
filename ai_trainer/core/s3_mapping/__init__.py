"""Common Skeleton 매핑 — 공정 3.

MediaPipe 33관절 -> 프로젝트 공통 18관절(common_skeleton의 정의)로 변환하고
저신뢰 관절을 freeze한다(pose_bridge). 2D/3D 두 경로가 같은 규칙을 쓴다.
시간축 필터는 두지 않는다 — MediaPipe 추정값을 그대로 넘긴다."""
