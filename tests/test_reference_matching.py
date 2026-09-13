"""3개 난이도 Reference any-match 판정과 Online 세션 연결 테스트."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.online_dtw import OnlineSquatSession
from ai_trainer.reference_levels import DIFFICULTY_LEVELS, NORMAL_CLASS, REFERENCE_CLASSES
from ai_trainer.reference_matching import (
    BINARY_DISTANCE_FEATURE_NAMES,
    INDETERMINATE_CLASS,
    UNSTABLE_CLASS,
    decide_reference_match,
)


ERROR_CLASSES = tuple(cls for cls in REFERENCE_CLASSES if cls != NORMAL_CLASS)
WEIGHTS_CFG = {"default_profile": "E_full_uniform", "uniform_weights": {}}


def _comparison(distance: float) -> dict:
    return {
        "min_distance": float(distance),
        "mean_topk_distance": float(distance),
        "best_detail": {"per_feature_contrib": {"controlled": float(distance)}},
    }


def _controlled_results(
    normal_distances: dict[str, float] | None = None,
) -> dict[str, dict[str, dict]]:
    normal_distances = normal_distances or {
        "초급": 0.8,
        "중급": 0.7,
        "고급": 0.9,
    }
    out = {}
    for level_index, level in enumerate(DIFFICULTY_LEVELS):
        out[level] = {NORMAL_CLASS: _comparison(normal_distances[level])}
        for class_index, class_label in enumerate(ERROR_CLASSES):
            out[level][class_label] = _comparison(
                1.0 + 0.1 * level_index + 0.01 * class_index
            )
    return out


def _session_db(
    distances: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, list[dict]]]:
    if distances is None:
        distances = {
            level: {
                class_label: 0.5 + 0.1 * level_index + 0.01 * class_index
                for class_index, class_label in enumerate(REFERENCE_CLASSES)
            }
            for level_index, level in enumerate(DIFFICULTY_LEVELS)
        }

    return {
        level: {
            class_label: [
                {
                    "feat": {"controlled": np.array([distances[level][class_label]])},
                    "bounds": {
                        "준비": [0, 1],
                        "하강": [0, 1],
                        "최저점": [0, 1],
                        "상승": [0, 1],
                        "종료": [0, 1],
                    },
                    "meta": {
                        "difficulty_level": level,
                        "class_label": class_label,
                    },
                    "controlled_distance": distances[level][class_label],
                }
            ]
            for class_label in REFERENCE_CLASSES
        }
        for level in DIFFICULTY_LEVELS
    }


def _session(db: dict | None = None, score_calib: dict | None = None) -> OnlineSquatSession:
    return OnlineSquatSession(
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        db_operational=db or _session_db(),
        weights_cfg=WEIGHTS_CFG,
        score_calib=score_calib,
    )


class ReferenceMatchDecisionTests(unittest.TestCase):
    def test_any_level_normal_match_uses_best_normal_level(self) -> None:
        results = _controlled_results({"초급": 0.8, "중급": 0.4, "고급": 0.7})
        # 오류 Reference가 전역적으로 더 가까워도, 보정 threshold를 통과한
        # 정상 Reference가 하나라도 있으면 any-level 정상이다.
        results["고급"][ERROR_CLASSES[0]] = _comparison(0.1)

        decision = decide_reference_match(
            results,
            score_calibration={
                "lo": 0.2,
                "hi": 1.2,
                "normal_distance_threshold": 0.5,
            },
        )

        self.assertEqual(decision.predicted_class, NORMAL_CLASS)
        self.assertEqual(decision.matched_level, "중급")
        self.assertAlmostEqual(decision.match_rate, 80.0)
        self.assertEqual(decision.normal_distance_by_level["중급"], 0.4)

    def test_threshold_miss_selects_closest_error_across_all_levels(self) -> None:
        results = _controlled_results({"초급": 0.6, "중급": 0.7, "고급": 0.8})
        results["중급"][ERROR_CLASSES[0]] = _comparison(0.2)
        results["고급"][ERROR_CLASSES[-1]] = _comparison(0.1)

        decision = decide_reference_match(
            results,
            score_calibration={
                "lo": 0.0,
                "hi": 1.0,
                "normal_distance_threshold": 0.5,
            },
        )

        self.assertEqual(decision.predicted_class, ERROR_CLASSES[-1])
        self.assertEqual(decision.matched_level, "고급")
        self.assertIs(decision.selected_result, results["고급"][ERROR_CLASSES[-1]])

    def test_without_calibration_uses_within_level_relative_normal_match(self) -> None:
        results = _controlled_results({"초급": 0.3, "중급": 0.8, "고급": 0.9})
        for class_label in ERROR_CLASSES:
            results["초급"][class_label] = _comparison(0.4)
        results["중급"][ERROR_CLASSES[0]] = _comparison(0.2)
        results["고급"][ERROR_CLASSES[0]] = _comparison(0.1)

        decision = decide_reference_match(results)

        self.assertEqual(decision.predicted_class, NORMAL_CLASS)
        self.assertEqual(decision.matched_level, "초급")
        self.assertIsNone(decision.match_rate)
        self.assertTrue(all(value is None for value in decision.normal_match_by_level.values()))

    def test_missing_level_or_class_is_rejected(self) -> None:
        missing_level = _controlled_results()
        missing_level.pop("고급")
        with self.assertRaisesRegex(ValueError, "난이도가 없습니다.*고급"):
            decide_reference_match(missing_level)

        missing_class = _controlled_results()
        missing_class["중급"].pop(ERROR_CLASSES[0])
        with self.assertRaisesRegex(ValueError, "중급.*클래스가 없습니다"):
            decide_reference_match(missing_class)

    def test_default_normal_match_threshold_is_sixty_percent(self) -> None:
        calibration_without_threshold = {"lo": 0.0, "hi": 1.0}

        passing = decide_reference_match(
            _controlled_results({level: 0.4 for level in DIFFICULTY_LEVELS}),
            score_calibration=calibration_without_threshold,
        )
        failing_results = _controlled_results(
            {level: 0.41 for level in DIFFICULTY_LEVELS}
        )
        failing_results["고급"][ERROR_CLASSES[0]] = _comparison(0.1)
        failing = decide_reference_match(
            failing_results,
            score_calibration=calibration_without_threshold,
        )

        self.assertEqual(passing.predicted_class, NORMAL_CLASS)
        self.assertAlmostEqual(passing.match_rate, 60.0)
        self.assertNotEqual(failing.predicted_class, NORMAL_CLASS)
        self.assertAlmostEqual(failing.match_rate, 59.0)

    def test_far_from_every_error_reference_is_unstable(self) -> None:
        decision = decide_reference_match(
            _controlled_results({level: 0.8 for level in DIFFICULTY_LEVELS}),
            score_calibration={
                "lo": 0.0,
                "hi": 2.0,
                "normal_distance_threshold": 0.5,
                "error_distance_thresholds": {
                    class_label: 0.4 for class_label in ERROR_CLASSES
                },
            },
        )
        self.assertEqual(decision.predicted_class, UNSTABLE_CLASS)
        self.assertEqual(decision.decision_status, "unstable")

    def test_close_first_and_second_error_is_indeterminate(self) -> None:
        results = _controlled_results({level: 0.8 for level in DIFFICULTY_LEVELS})
        results["초급"][ERROR_CLASSES[0]] = _comparison(0.20)
        results["중급"][ERROR_CLASSES[1]] = _comparison(0.205)
        decision = decide_reference_match(
            results,
            score_calibration={
                "lo": 0.0,
                "hi": 2.0,
                "normal_distance_threshold": 0.5,
                "error_distance_thresholds": {
                    class_label: 1.0 for class_label in ERROR_CLASSES
                },
                "min_error_relative_margin": 0.05,
            },
        )
        self.assertEqual(decision.predicted_class, INDETERMINATE_CLASS)
        self.assertEqual(decision.decision_status, "indeterminate")

    def test_binary_meta_classifier_controls_first_stage(self) -> None:
        size = len(BINARY_DISTANCE_FEATURE_NAMES)
        decision = decide_reference_match(
            _controlled_results({level: 0.8 for level in DIFFICULTY_LEVELS}),
            score_calibration={
                "lo": 0.0,
                "hi": 2.0,
                "normal_distance_threshold": 0.1,
                "binary_classifier": {
                    "feature_names": list(BINARY_DISTANCE_FEATURE_NAMES),
                    "impute": [0.0] * size,
                    "mean": [0.0] * size,
                    "scale": [1.0] * size,
                    "intercept": 10.0,
                    "coefficients": [0.0] * size,
                    "probability_threshold": 0.5,
                },
            },
        )
        self.assertEqual(decision.predicted_class, NORMAL_CLASS)
        self.assertGreater(decision.normal_probability, 0.99)

    def test_error_distance_normalization_controls_second_stage(self) -> None:
        results = _controlled_results({level: 0.8 for level in DIFFICULTY_LEVELS})
        results["초급"][ERROR_CLASSES[0]] = _comparison(0.20)
        results["초급"][ERROR_CLASSES[1]] = _comparison(0.30)
        results["초급"][ERROR_CLASSES[2]] = _comparison(0.40)
        decision = decide_reference_match(
            results,
            score_calibration={
                "lo": 0.0,
                "hi": 2.0,
                "normal_distance_threshold": 0.5,
                "error_distance_normalization": {
                    ERROR_CLASSES[0]: {"median": 0.0, "iqr": 0.10},
                    ERROR_CLASSES[1]: {"median": 0.29, "iqr": 0.10},
                    ERROR_CLASSES[2]: {"median": 0.0, "iqr": 0.10},
                },
            },
        )
        self.assertEqual(decision.predicted_class, ERROR_CLASSES[1])
        self.assertGreater(decision.error_z_margin, 0.0)

    def test_2d_heel_evidence_can_override_or_suppress_heel_result(self) -> None:
        results = _controlled_results({level: 0.2 for level in DIFFICULTY_LEVELS})
        results["초급"][ERROR_CLASSES[0]] = _comparison(0.1)
        calibration = {
            "lo": 0.0,
            "hi": 2.0,
            "normal_distance_threshold": 0.5,
        }
        lifted = decide_reference_match(
            results,
            score_calibration=calibration,
            heel_contact_2d=False,
        )
        stable_results = _controlled_results(
            {level: 0.8 for level in DIFFICULTY_LEVELS}
        )
        stable_results["초급"][ERROR_CLASSES[0]] = _comparison(0.1)
        stable = decide_reference_match(
            stable_results,
            score_calibration=calibration,
            heel_contact_2d=True,
        )
        self.assertEqual(lifted.predicted_class, ERROR_CLASSES[0])
        self.assertEqual(stable.predicted_class, INDETERMINATE_CLASS)
        self.assertEqual(stable.rejection_reason, "heel_error_not_supported_by_2d")


class OnlineSquatSessionReferenceTests(unittest.TestCase):
    def test_session_rejects_missing_level_class_and_mismatched_metadata(self) -> None:
        missing_level = _session_db()
        missing_level.pop("고급")
        with self.assertRaisesRegex(ValueError, "난이도가 없습니다.*고급"):
            _session(missing_level)

        empty_class = _session_db()
        empty_class["중급"][ERROR_CLASSES[0]] = []
        with self.assertRaisesRegex(ValueError, "level=중급, class="):
            _session(empty_class)

        wrong_meta = _session_db()
        wrong_meta["초급"][NORMAL_CLASS][0]["meta"]["difficulty_level"] = "고급"
        with self.assertRaisesRegex(ValueError, "metadata가 일치하지 않습니다"):
            _session(wrong_meta)

    def test_finalize_rep_records_all_level_distances_and_match_fields(self) -> None:
        distances = {
            level: {
                class_label: 0.9 + 0.1 * level_index + 0.01 * class_index
                for class_index, class_label in enumerate(REFERENCE_CLASSES)
            }
            for level_index, level in enumerate(DIFFICULTY_LEVELS)
        }
        distances["중급"][NORMAL_CLASS] = 0.4
        distances["고급"][ERROR_CLASSES[0]] = 0.1
        session = _session(
            _session_db(distances),
            score_calib={
                "lo": 0.2,
                "hi": 1.2,
                "normal_distance_threshold": 0.5,
            },
        )
        session.emit_offset = 0
        session.rep_start_idx = 0
        session.aligned_seq = [np.zeros((18, 3), dtype=np.float32) for _ in range(3)]

        def fake_multi_reference(*args, **kwargs):
            medoids = args[2]
            distance = medoids[0]["controlled_distance"]
            return _comparison(distance)

        with (
            patch("ai_trainer.online_dtw.extract_all_features", return_value={}),
            patch(
                "ai_trainer.online_dtw.multi_reference_distance",
                side_effect=fake_multi_reference,
            ) as compare,
        ):
            session._finalize_rep(2)

        self.assertEqual(compare.call_count, len(DIFFICULTY_LEVELS) * len(REFERENCE_CLASSES))
        result = session.completed_reps[-1]
        self.assertEqual(result.predicted_class, NORMAL_CLASS)
        self.assertEqual(result.matched_level, "중급")
        self.assertAlmostEqual(result.match_rate, 80.0)
        self.assertEqual(result.score_vs_normal, result.match_rate)
        self.assertEqual(result.raw_distance_by_level, distances)
        self.assertEqual(
            set(result.normal_distance_by_level),
            set(DIFFICULTY_LEVELS),
        )
        self.assertEqual(
            result.raw_distance_by_class[ERROR_CLASSES[0]],
            distances["고급"][ERROR_CLASSES[0]],
        )

    def test_partial_distance_reports_any_level_decision_and_nested_distances(self) -> None:
        distances = {
            level: {
                class_label: 0.7 + 0.1 * level_index + 0.01 * class_index
                for class_index, class_label in enumerate(REFERENCE_CLASSES)
            }
            for level_index, level in enumerate(DIFFICULTY_LEVELS)
        }
        distances["초급"][NORMAL_CLASS] = 0.3
        for class_label in ERROR_CLASSES:
            distances["초급"][class_label] = 0.4
        distances["고급"][ERROR_CLASSES[0]] = 0.1

        session = _session(_session_db(distances))
        session.state = "descend"
        session.current_phase_start = 0
        session.emit_offset = 0
        session.aligned_seq = [np.zeros((18, 3), dtype=np.float32) for _ in range(3)]

        def fake_cost(query_feat, reference_feat, weights, config):
            n_query = next(iter(query_feat.values())).shape[0]
            distance = float(next(iter(reference_feat.values()))[0])
            return np.full((n_query, 1), distance), {}

        with (
            patch(
                "ai_trainer.online_dtw.extract_all_features",
                return_value={"controlled": np.ones(3)},
            ),
            patch(
                "ai_trainer.dtw_compare.weighted_frame_cost_matrix",
                side_effect=fake_cost,
            ) as frame_cost,
        ):
            partial = session._partial_online_distance(2)

        self.assertIsNotNone(partial)
        self.assertEqual(frame_cost.call_count, len(DIFFICULTY_LEVELS) * len(REFERENCE_CLASSES))
        self.assertEqual(partial["predicted_class"], NORMAL_CLASS)
        self.assertEqual(partial["matched_level"], "초급")
        for level in DIFFICULTY_LEVELS:
            for class_label in REFERENCE_CLASSES:
                self.assertAlmostEqual(
                    partial["distance_by_level"][level][class_label],
                    distances[level][class_label],
                )
        self.assertEqual(
            partial["distance_by_class"][ERROR_CLASSES[0]],
            partial["distance_by_level"]["고급"][ERROR_CLASSES[0]],
        )

    def test_partial_distance_uses_phase_specific_normal_threshold(self) -> None:
        distances = {
            level: {
                class_label: 0.4 + 0.1 * level_index + 0.01 * class_index
                for class_index, class_label in enumerate(REFERENCE_CLASSES)
            }
            for level_index, level in enumerate(DIFFICULTY_LEVELS)
        }
        distances["초급"][NORMAL_CLASS] = 0.3
        distances["초급"][ERROR_CLASSES[0]] = 0.2
        weights_cfg = {
            **WEIGHTS_CFG,
            "partial_normal_distance_thresholds": {"하강": 0.25},
        }
        session = OnlineSquatSession(
            model=torch.nn.Identity(),
            device=torch.device("cpu"),
            db_operational=_session_db(distances),
            weights_cfg=weights_cfg,
            score_calib={
                "lo": 0.0,
                "hi": 1.0,
                "normal_distance_threshold": 0.5,
            },
        )
        session.state = "descend"
        session.current_phase_start = 0
        session.emit_offset = 0
        session.aligned_seq = [
            np.zeros((18, 3), dtype=np.float32) for _ in range(3)
        ]

        def fake_cost(query_feat, reference_feat, weights, config):
            n_query = next(iter(query_feat.values())).shape[0]
            distance = float(next(iter(reference_feat.values()))[0])
            return np.full((n_query, 1), distance), {}

        with (
            patch(
                "ai_trainer.online_dtw.extract_all_features",
                return_value={"controlled": np.ones(3)},
            ),
            patch(
                "ai_trainer.dtw_compare.weighted_frame_cost_matrix",
                side_effect=fake_cost,
            ),
        ):
            partial = session._partial_online_distance(2)

        self.assertIsNotNone(partial)
        self.assertEqual(partial["normal_distance_threshold"], 0.25)
        self.assertNotEqual(partial["predicted_class"], NORMAL_CLASS)


if __name__ == "__main__":
    unittest.main()
