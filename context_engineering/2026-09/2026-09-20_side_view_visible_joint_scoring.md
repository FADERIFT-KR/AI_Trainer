# Side-view scoring from visible joints

In a side view, the squat judge now uses the **camera-near shoulder, hip, knee, ankle, heel, and big toe**. The camera-far limb and both elbows/wrists have zero weight in the normal-template DTW gate and are zeroed before the side-trained MT-ST-GCN. The derived pelvis and neck nodes are also excluded from side scoring because they can depend on the far side. Left and right sides use separate actor-calibrated DTW thresholds. The frontal scoring path and its artifacts are unchanged.

The side pose is centered at the near hip and scaled by the near leg length. Repetition phase height uses the near hip-to-ankle vertical distance, not the far ankle. Live joint-angle feedback reports only the near knee, hip, and ankle, and its reference frame is matched using the near-side template. The old whole-body DTW diagnostic is not used for side decisions or side explanations. Camera orientation initialization still needs both hips in the first standing frames; this is a structural coordinate-frame estimate, not a frame-by-frame far-leg score.

Run `python scripts/train_side_view_squat.py` after the frontal two-stage training to create:

- `output/two_stage_squat/side_view_gates.npz`
- `output/two_stage_squat/side_view_mt_stgcn.pt`
- `output/two_stage_squat/side_view_report.json`

The side model is trained with the same masks used in inference. If the side artifacts are missing, the application returns `판정 불확실` instead of silently applying the unmasked frontal/legacy classifier to a side view.

## Evidence and limits

On the existing actor-disjoint **AI Hub 3D CSV** holdout (66 repetitions per simulated side view), the left-side gate/model has binary accuracy 0.879, normal recall 0.958, and error false-pass rate 0.167; the right-side values are 0.848, 1.000, and 0.238. The full-joint frontal model had 0.924 binary accuracy and 0.119 error false-pass on that split. Therefore removing hidden-side evidence **did not demonstrate an accuracy gain** on the available labeled data, although it prevents a hidden-side spike from changing the side input. The reused holdout was consulted during earlier development and is exploratory, not a sealed test; real-webcam error rates are unknown.

Replaying the newer `squat_session_20260917_213840` JSONL through the full side-view streaming path still detects three repetitions on each side. Their near-side DTW distances remain above the calibrated thresholds (left 0.251–0.370 vs 0.093; right 0.554–0.629 vs 0.099). The side model suggests errors, while the recorded view-condition rules do not repeatedly corroborate them. The conservative final label therefore remains `판정 불확실`. This replay uses saved 3D estimates and cannot prove that the visible joints themselves are geometrically correct.

To establish a true accuracy improvement, collect independently labeled webcam squats (normal and each error type) from multiple people at all three views, then compare the original and visible-joint systems on an untouched actor holdout. Do not tune a normal threshold from unlabeled recordings alone.
