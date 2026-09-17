"""Evaluate manually labelled live heel-shadow rows. Blank ground truth is excluded."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from ai_trainer.heel_policy_shadow import classification_metrics
from ai_trainer.live_heel_shadow import SUPPORTED_GROUND_TRUTH


def evaluate(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        annotated = path / "annotation_template.csv"
        path = annotated if annotated.exists() else path / "reps.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    valid = [row for row in rows if (row.get("ground_truth_label") or "").strip()]
    unsupported = sorted({row["ground_truth_label"].strip() for row in valid
                          if row["ground_truth_label"].strip() not in SUPPORTED_GROUND_TRUTH})
    if unsupported: raise ValueError(f"unsupported ground-truth labels: {unsupported}")
    binary = [row for row in valid if row["ground_truth_label"].strip() in ("정상", "발뒤꿈치오류")]
    result = {"input_rows": len(rows), "manually_labelled_n": len(valid), "evaluated_binary_n": len(binary),
              "excluded_unlabelled_n": len(rows) - len(valid), "excluded_nonbinary_n": len(valid) - len(binary)}
    for name, field in (("production", "production_final"), ("policy_b", "shadow_b_final"),
                        ("policy_d", "shadow_d_final")):
        result[name] = classification_metrics([
            {"truth": row["ground_truth_label"].strip(), "result": row[field]} for row in binary
        ]) if binary else None
    output = path.parent / "manual_label_evaluation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", help="session directory, annotation_template.csv, or reps.csv")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.session), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
