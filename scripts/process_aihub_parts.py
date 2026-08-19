"""Inspect, merge, or stream-extract AI Hub .part archives on Windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ai_trainer.reference.parts import (  # noqa: E402
    PartSetError,
    discover_part_set,
    extract_tar,
    list_tar_members,
    merge_parts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and process AI Hub archive.part<offset> files without WSL. "
            "With no action, only a safe read-only verification is performed."
        )
    )
    parser.add_argument("--parts", required=True, type=Path, help="Part directory or one part file")
    parser.add_argument(
        "--archive",
        help="Original archive name when the directory contains more than one set",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verify", action="store_true", help="Validate only (default)")
    actions.add_argument(
        "--list",
        nargs="?",
        const=30,
        type=int,
        metavar="LIMIT",
        help="List the first LIMIT TAR members without merging (default: 30)",
    )
    actions.add_argument("--merge", type=Path, metavar="FILE", help="Create the original archive")
    actions.add_argument(
        "--extract",
        type=Path,
        metavar="DIRECTORY",
        help="Stream-extract split TAR parts without creating the merged TAR",
    )
    parser.add_argument(
        "--member",
        action="append",
        default=None,
        help="Exact TAR member to extract; repeat for multiple members",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.member and args.extract is None:
        print("--member requires --extract", file=sys.stderr)
        return 2
    try:
        part_set = discover_part_set(args.parts, args.archive)
        print(f"Archive: {part_set.archive_name}")
        print(f"Parts: {len(part_set.parts)}")
        print(f"Logical size: {part_set.total_size:,} bytes")
        print("Continuity and archive markers: OK")

        if args.list is not None:
            for name in list_tar_members(part_set, args.list):
                print(name)
        elif args.merge is not None:
            output = merge_parts(part_set, args.merge, overwrite=args.overwrite)
            print(f"Merged archive: {output}")
        elif args.extract is not None:
            destination = extract_tar(
                part_set,
                args.extract,
                overwrite=args.overwrite,
                members=args.member,
            )
            print(f"Extracted TAR destination: {destination}")
    except (PartSetError, OSError) as error:
        print(f"Part processing failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
