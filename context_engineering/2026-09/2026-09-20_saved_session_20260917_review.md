# Saved-session skeleton and judgment review (2026-09-17)

Source: the two read-only recordings in `C:\Users\user\Desktop\TP\saved\squat_session_20260917_205306` and `squat_session_20260917_213840`. Each has three front, three left, and three right squats. The videos and original JSONL files were not altered.

## Skeleton tracking

The newer session has a valid framing gate on all 480 recorded frames (145 front, 175 left, 160 right). The visually disruptive jumps occur mainly in the camera-far wrist, not in the left knee. `scripts/analyze_recorded_session.py` reports no isolated world-landmark knee jumps above its 0.12 m review threshold. Examples from the displayed, body-normalized 3D skeleton:

| View | Video frame / time | Far-wrist visibility | Old frame-to-frame displacement |
| --- | ---: | ---: | ---: |
| Left | 77 / 8.12 s | 0.061 | 0.456 |
| Left | 170 / 19.42 s | 0.206 | 0.565 |
| Right | 29 / 3.55 s | 0.205 | 0.698 |
| Right | 85 / 10.43 s | 0.311 | 0.670 |
| Right | 89 / 10.96 s | 0.265 | 0.627 |

The 3D bridge could pass raw estimates for a landmark that had **never** been confidently observed, even while marking it frozen. Side views make this common for the far arm. The display pipeline now mirrors the confidently visible arm about the body for the occluded far elbow/wrist, blends the transition as visibility improves, and caps an unsupported lateral 3D leap of the visible wrist when its 2D image position barely moves. These changes affect the display pose only; DTW, phase detection, and graph-classifier inputs retain their original 3D coordinates.

Offline replay using the saved landmarks reduced the maximum far-wrist frame displacement from **0.565 to 0.186** on the left and **0.698 to 0.227** on the right. At the listed frames the changes were 0.456→0.157, 0.565→0.040, 0.698→0.043, 0.670→0.227, and 0.627→0.071 respectively. The corrected comparison videos are in `output/replay_saved/left_corrected_v2_20260917_213840.mp4` and `right_corrected_v2_20260917_213840.mp4`. They reuse stored landmarks rather than rerunning MediaPipe; this validates the display reconstruction, **not** new model inference or live-camera performance. Mirroring assumes roughly symmetric arm placement and is deliberately not used for scoring.

## Normal-vs-error criteria

The current DTW threshold is **0.096942**, fitted on AI Hub skeleton data (28 normal and 40 error calibration repetitions). Its actor-disjoint dataset validation had 100% normal recall and 11.9% error false-pass on 66 repetitions, but this does not establish accuracy on webcam MediaPipe world landmarks. In the newer session, all nine DTW distances are **0.337–0.535**, roughly 3.5–5.5 times the threshold. Consequently all nine repetitions are sent to the error classifier. However, the independently fitted view-condition rules yield **no error support ≥0.60** for any of those nine repetitions; the final decision is appropriately **“판정 불확실”** rather than a confirmed error.

On the newer side-view videos, the camera-near hip image point is below the camera-near knee at the bottom of all six repetitions (hip-minus-knee image vertical distance / visible leg length: left **0.073–0.159**, right **0.075–0.120**). This is evidence against calling these repetitions *obviously too shallow*, although image landmarks are not a clinical ground truth and foot contact cannot be certified from these recordings alone. Thus the model's repeated `엉덩이하방오류` hypotheses are not sufficiently supported for a definitive label.

The older session's right view retained only **107/182** frames under the then-current framing gate. It reported `엉덩이하방오류` with rule support 1.0 on two repetitions, but the discarded squat-bottom frames make that verdict unreliable. The newer recording has 160/160 right-view frames accepted after the framing fix, and its final decision is uncertain. Old and new session outcomes should not be treated as evidence that the participant's form changed.

**Conclusion:** the dataset-derived normal gate is not yet calibrated for these real-camera recordings. Do not raise the threshold from nine unlabeled repetitions or turn the uncertain decision into normal; that could increase false normal passes. Keep `판정 불확실` when DTW and view rules disagree, and obtain independently labeled normal/error webcam repetitions from multiple people and all three views before recalibrating the threshold, validating per-view rules, and reporting sensitivity/specificity on a held-out set. Per-repetition error names currently displayed by the model are hypotheses, not confirmed judgments.
