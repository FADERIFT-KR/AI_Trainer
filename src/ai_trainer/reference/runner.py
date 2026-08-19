"""Friendly Python/IDE entry point with a no-data demo fallback."""

from __future__ import annotations

import sys
from pathlib import Path

from .cli import main as build_main
from .demo import create_demo_preview


def _has_downloaded_sources(input_root: Path) -> bool:
    if not input_root.is_dir():
        return False
    has_json = any(path.suffix.casefold() == ".json" for path in input_root.rglob("*"))
    if not has_json:
        return False
    return any(path.suffix.casefold() == ".csv" for path in input_root.rglob("*"))


def main(
    argv: list[str] | None = None,
    *,
    project_root: str | Path | None = None,
) -> int:
    """Run a real build when arguments/data exist, otherwise render the demo."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()

    if arguments:
        if arguments == ["--demo"]:
            output = create_demo_preview(
                root / "data" / "reference" / "demo" / "reference_preview.png"
            )
            print(f"Synthetic skeleton preview created: {output}")
            return 0
        return build_main(arguments)

    input_root = root / "data" / "raw" / "aihub_crossfit"
    if _has_downloaded_sources(input_root):
        print(f"AI Hub source files detected: {input_root}")
        return build_main(
            [
                "--input",
                str(input_root),
                "--output",
                str(root / "data" / "reference" / "air_squat"),
            ]
        )

    output = create_demo_preview(
        root / "data" / "reference" / "demo" / "reference_preview.png"
    )
    print("AI Hub source files were not found; generated the safe synthetic example instead.")
    print(f"Skeleton preview: {output}")
    return 0
