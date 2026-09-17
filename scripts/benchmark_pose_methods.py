"""Run the research-only Reference Leave-One-Out Benchmark."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trainer.aihub_zip import AiHubZip
from ai_trainer.common_skeleton import to_common_skeleton
from ai_trainer.offline_pose_benchmark import OfflinePoseBenchmark
from scripts.build_reference_db import TL_ZIP, VL_ZIP


def load_operational_2d(root: Path) -> dict:
    entries = json.loads((root/"output"/"reference_db"/"manifest.json").read_text(encoding="utf-8"))["entries"]
    archives = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}
    result = {}
    try:
        for entry in entries:
            if entry["tier"] != "operational":
                continue
            archive = archives[entry["origin_zip"]]
            candidates = archive.find_sequences(error_type=entry["class_label"],
                level=entry["difficulty_level"], actor=entry["actor_id"], rep=entry["repetition_id"])
            if len(candidates) != 1:
                raise ValueError(f"2D source match count={len(candidates)} for {entry['medoid_id']}")
            _, points = archive.read_2d(candidates[0], 1)
            start, end = entry["frame_range"]
            result[entry["medoid_id"]] = to_common_skeleton(points[start:end+1])
    finally:
        for archive in archives.values():
            archive.close()
    return result


def main() -> int:
    raw2d = load_operational_2d(ROOT)
    benchmark = OfflinePoseBenchmark(ROOT, raw2d)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT/"output"/"diagnostics"/"offline_benchmark"/stamp
    summary, _ = benchmark.run(output)
    print(f"[OFFLINE BENCHMARK] {output}")
    for method, result in summary["results"].items():
        metrics = result["metrics"]
        print(f"{method}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
