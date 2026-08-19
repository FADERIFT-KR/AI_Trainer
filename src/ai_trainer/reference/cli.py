"""Command-line entry point for building air-squat reference data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import BuildConfig, BuildError, build_reference


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized, phase-aligned normal air-squat reference data from "
            "AI Hub dataset 71422 JSON and 3-D CSV files."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Extracted AI Hub dataset root")
    parser.add_argument("--output", required=True, type=Path, help="Reference artifact directory")
    parser.add_argument(
        "--pairs-manifest",
        type=Path,
        help="CSV mapping annotation_json to keypoints_3d_csv (recommended)",
    )
    parser.add_argument("--target-frames", type=int, default=101)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-frames", type=int, default=15)
    parser.add_argument("--min-flexion-deg", type=float, default=25.0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.95)
    parser.add_argument("--max-missing-gap-frames", type=int, default=2)
    parser.add_argument("--min-repetitions", type=int, default=3)
    parser.add_argument("--outlier-mad-threshold", type=float, default=4.0)
    parser.add_argument(
        "--no-repetitions",
        action="store_true",
        help="Do not retain the per-repetition NPZ; write only the aggregate template",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create the final three-pose skeleton preview PNG",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    config = BuildConfig(
        input_root=args.input,
        output_dir=args.output,
        pairs_manifest=args.pairs_manifest,
        target_frames=args.target_frames,
        fps=args.fps,
        min_frames=args.min_frames,
        min_flexion_deg=args.min_flexion_deg,
        min_valid_ratio=args.min_valid_ratio,
        max_missing_gap_frames=args.max_missing_gap_frames,
        min_repetitions=args.min_repetitions,
        outlier_mad_threshold=args.outlier_mad_threshold,
        include_repetitions=not args.no_repetitions,
        include_plot=not args.no_plot,
        overwrite=args.overwrite,
    )
    try:
        result = build_reference(config)
    except (BuildError, ValueError, OSError) as error:
        print(f"Reference build failed: {error}", file=sys.stderr)
        return 2
    print(
        f"Built {result.accepted_repetitions}/{result.total_repetitions} accepted "
        f"repetitions in {result.output_dir}"
    )
    if result.skipped_sources:
        print(f"Skipped source intervals/files: {result.skipped_sources}")
    if result.preview_path is not None:
        print(f"Skeleton preview: {result.preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
