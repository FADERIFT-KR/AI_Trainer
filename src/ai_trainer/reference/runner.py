"""Friendly Python/IDE entry point with a no-data demo fallback."""

from __future__ import annotations

import sys
from pathlib import Path

from ai_trainer.runtime_paths import configured_dataset_path, reference_output_dir

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

    selected_dataset = configured_dataset_path()
    input_root = selected_dataset or root / "data" / "raw" / "aihub_crossfit"
    if _has_downloaded_sources(input_root):
        print(f"AI Hub source files detected: {input_root}")
        output_root = (
            reference_output_dir()
            if selected_dataset is not None
            else root / "data" / "reference" / "air_squat"
        )
        return build_main(
            [
                "--input",
                str(input_root),
                "--output",
                str(output_root),
            ]
        )

    output = create_demo_preview(
        root / "data" / "reference" / "demo" / "reference_preview.png"
    )
    print("AI Hub source files were not found; generated the safe synthetic example instead.")
    print(f"Skeleton preview: {output}")
    return 0
