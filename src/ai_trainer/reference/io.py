"""Read and pair AI Hub annotation JSON and 3-D keypoint CSV files."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .schema import COMMON_JOINTS, is_normal_air_squat


class DataFormatError(ValueError):
    """Raised when a source file does not match the documented AI Hub schema."""


@dataclass(frozen=True)
class SourcePair:
    annotation_json: Path
    keypoints_csv: Path
    pairing_method: str
    annotation_index: int | None = None
    sample_id: str | None = None


@dataclass
class DiscoveryResult:
    pairs: list[SourcePair] = field(default_factory=list)
    normal_air_squat_json_count: int = 0
    three_dimensional_csv_count: int = 0
    unmatched_json: list[str] = field(default_factory=list)
    ambiguous_json: list[str] = field(default_factory=list)
    unreadable_files: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class SkeletonSequence:
    points: np.ndarray
    frame_numbers: np.ndarray
    image_filenames: tuple[str, ...]


@dataclass(frozen=True)
class SlicedSequence:
    points: np.ndarray
    row_start: int
    row_end: int
    source_frame_start: int | None
    source_frame_end: int | None


_ENCODINGS = ("utf-8-sig", "utf-8", "cp949")
_GENERIC_STEM_PARTS = re.compile(
    r"(?i)(?:annotations?|labels?|keypoints?|skeleton|pose|coordinates?|3d)"
)


def _read_text(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise DataFormatError(f"Cannot decode {path}: {last_error}")


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_text(path))
    except json.JSONDecodeError as error:
        raise DataFormatError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataFormatError(f"Metadata root must be an object: {path}")
    annotations = value.get("annotations")
    if not isinstance(annotations, list):
        raise DataFormatError(f"Metadata must contain an annotations list: {path}")
    return value


def iter_normal_air_squat_annotations(
    metadata: Mapping[str, Any], annotation_index: int | None = None
) -> Iterable[tuple[int, Mapping[str, Any]]]:
    annotations = metadata.get("annotations", [])
    for index, annotation in enumerate(annotations):
        if annotation_index is not None and index != annotation_index:
            continue
        if isinstance(annotation, Mapping) and is_normal_air_squat(annotation):
            yield index, annotation


def _normalized_column(name: Any) -> str:
    value = unicodedata.normalize("NFC", str(name).lstrip("\ufeff"))
    return re.sub(r"[\s\-]+", "", value).casefold()


def _read_csv(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, nrows=nrows)
        except UnicodeDecodeError as error:
            last_error = error
        except pd.errors.EmptyDataError as error:
            raise DataFormatError(f"CSV is empty: {path}") from error
        except pd.errors.ParserError as error:
            raise DataFormatError(f"Malformed CSV {path}: {error}") from error
    raise DataFormatError(f"Cannot decode CSV {path}: {last_error}")


def is_3d_keypoint_csv(path: Path) -> bool:
    try:
        header = _read_csv(path, nrows=0)
    except (OSError, DataFormatError, pd.errors.ParserError):
        return False
    columns = {_normalized_column(column) for column in header.columns}
    probes = ("nose_z", "lhip_z", "rhip_z", "lankle_z", "rankle_z")
    return all(_normalized_column(probe) in columns for probe in probes)


def _frame_number(filename: str) -> int | None:
    numbers = re.findall(r"\d+", Path(filename).stem)
    return int(numbers[-1]) if numbers else None


def read_3d_keypoints(path: Path) -> SkeletonSequence:
    frame = _read_csv(path)
    normalized_to_original: dict[str, str] = {}
    for column in frame.columns:
        normalized = _normalized_column(column)
        if normalized in normalized_to_original:
            raise DataFormatError(f"Duplicate normalized CSV column {column!r} in {path}")
        normalized_to_original[normalized] = str(column)

    joint_arrays: list[np.ndarray] = []
    missing: list[str] = []
    for joint in COMMON_JOINTS:
        coordinate_columns: list[str] = []
        for axis in "xyz":
            key = _normalized_column(f"{joint}_{axis}")
            original = normalized_to_original.get(key)
            if original is None:
                coordinate_columns = []
                missing.append(f"{joint}_{axis}")
                break
            coordinate_columns.append(original)
        if coordinate_columns:
            numeric = frame[coordinate_columns].apply(pd.to_numeric, errors="coerce")
            joint_arrays.append(numeric.to_numpy(dtype=np.float64))

    # Hip and Neck are documented columns, but accepting their midpoint fallback
    # makes the reader compatible with exports that retain only limb joints.
    if missing:
        recoverable = {"Hip_x", "Hip_y", "Hip_z", "Neck_x", "Neck_y", "Neck_z"}
        if not set(missing).issubset(recoverable):
            preview = ", ".join(missing[:8])
            raise DataFormatError(f"Missing required 3-D columns in {path}: {preview}")
        joint_arrays = []
        for joint in COMMON_JOINTS:
            keys = [_normalized_column(f"{joint}_{axis}") for axis in "xyz"]
            originals = [normalized_to_original.get(key) for key in keys]
            if all(originals):
                numeric = frame[list(originals)].apply(pd.to_numeric, errors="coerce")
                joint_arrays.append(numeric.to_numpy(dtype=np.float64))
            elif joint == "Hip":
                left = np.column_stack(
                    [
                        pd.to_numeric(frame[normalized_to_original[_normalized_column(f"LHip_{axis}")]], errors="coerce")
                        for axis in "xyz"
                    ]
                )
                right = np.column_stack(
                    [
                        pd.to_numeric(frame[normalized_to_original[_normalized_column(f"RHip_{axis}")]], errors="coerce")
                        for axis in "xyz"
                    ]
                )
                joint_arrays.append((left + right) / 2.0)
            elif joint == "Neck":
                left = np.column_stack(
                    [
                        pd.to_numeric(
                            frame[normalized_to_original[_normalized_column(f"LShoulder_{axis}")]],
                            errors="coerce",
                        )
                        for axis in "xyz"
                    ]
                )
                right = np.column_stack(
                    [
                        pd.to_numeric(
                            frame[normalized_to_original[_normalized_column(f"RShoulder_{axis}")]],
                            errors="coerce",
                        )
                        for axis in "xyz"
                    ]
                )
                joint_arrays.append((left + right) / 2.0)
            else:  # pragma: no cover - guarded by recoverable set above
                raise AssertionError(joint)

    points = np.stack(joint_arrays, axis=1)
    filename_column = normalized_to_original.get(_normalized_column("image_filename"))
    if filename_column is None:
        filenames = tuple("" for _ in range(len(frame)))
        frame_numbers = np.full(len(frame), -1, dtype=np.int64)
    else:
        filenames = tuple(frame[filename_column].fillna("").astype(str))
        parsed = [_frame_number(filename) for filename in filenames]
        if any(value is None for value in parsed):
            raise DataFormatError(f"Every image_filename must contain a frame number: {path}")
        frame_numbers = np.asarray([-1 if value is None else value for value in parsed], dtype=np.int64)
        valid_numbers = frame_numbers[frame_numbers >= 0]
        if valid_numbers.size > 1 and np.any(np.diff(valid_numbers) != 1):
            raise DataFormatError(
                f"image_filename frame IDs must be unique, increasing, and contiguous: {path}"
            )
    return SkeletonSequence(points=points, frame_numbers=frame_numbers, image_filenames=filenames)


def _optional_number(value: Any, cast: type[int] | type[float]) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        return cast(value)
    except (TypeError, ValueError) as error:
        raise DataFormatError(f"Invalid annotation bound {value!r}") from error


def slice_annotation(
    sequence: SkeletonSequence,
    annotation: Mapping[str, Any],
    *,
    fps: float,
) -> SlicedSequence:
    """Slice an inclusive AI Hub annotation range from a skeleton CSV."""

    start = _optional_number(annotation.get("start_frame"), int)
    end = _optional_number(annotation.get("end_frame"), int)
    if start is None:
        start_time = _optional_number(annotation.get("start_time"), float)
        start = 0 if start_time is None else int(round(float(start_time) * fps))
    if end is None:
        end_time = _optional_number(annotation.get("end_time"), float)
        end = len(sequence.points) - 1 if end_time is None else int(round(float(end_time) * fps))
    if start < 0 or end < start:
        raise DataFormatError(f"Invalid frame interval [{start}, {end}]")

    valid_ids = sequence.frame_numbers >= 0
    if valid_ids.any():
        mask = valid_ids & (sequence.frame_numbers >= start) & (sequence.frame_numbers <= end)
        matching = np.flatnonzero(mask)
        if matching.size:
            first, last = int(matching[0]), int(matching[-1])
            if not np.array_equal(matching, np.arange(first, last + 1)):
                raise DataFormatError("Annotation frame IDs are not contiguous in the 3-D CSV")
            return SlicedSequence(
                points=sequence.points[first : last + 1],
                row_start=first,
                row_end=last,
                source_frame_start=int(sequence.frame_numbers[first]),
                source_frame_end=int(sequence.frame_numbers[last]),
            )

    # Some exports omit image_filename or use names unrelated to source frame IDs.
    # Positional slicing is safe only when the documented bounds fit the CSV.
    if end < len(sequence.points):
        return SlicedSequence(
            points=sequence.points[start : end + 1],
            row_start=start,
            row_end=end,
            source_frame_start=start,
            source_frame_end=end,
        )
    if start >= 1 and end <= len(sequence.points):
        # Accommodate one-based frame numbers without silently clipping.
        return SlicedSequence(
            points=sequence.points[start - 1 : end],
            row_start=start - 1,
            row_end=end - 1,
            source_frame_start=start,
            source_frame_end=end,
        )
    raise DataFormatError(
        f"Annotation interval [{start}, {end}] does not match CSV with {len(sequence.points)} rows"
    )


def read_pair_manifest(path: Path, input_root: Path) -> list[SourcePair]:
    """Read explicit JSON-to-3D-CSV mappings, preferred for AI Hub archives."""

    text = _read_text(path)
    rows = list(csv.DictReader(text.splitlines()))
    required = {"annotation_json", "keypoints_3d_csv"}
    if not rows:
        raise DataFormatError(f"Pair manifest has no data rows: {path}")
    if not required.issubset(rows[0]):
        raise DataFormatError(
            f"Pair manifest needs columns: {', '.join(sorted(required))}"
        )

    def resolve(value: str) -> Path:
        candidate = Path(value)
        resolved = candidate.resolve() if candidate.is_absolute() else (input_root / candidate).resolve()
        try:
            resolved.relative_to(input_root.resolve())
        except ValueError as error:
            raise DataFormatError(
                f"Manifest path must stay inside the input root: {resolved}"
            ) from error
        return resolved

    pairs: list[SourcePair] = []
    for row_number, row in enumerate(rows, start=2):
        annotation_path = resolve(row["annotation_json"].strip())
        keypoint_path = resolve(row["keypoints_3d_csv"].strip())
        if not annotation_path.is_file():
            raise DataFormatError(f"Manifest row {row_number}: JSON not found: {annotation_path}")
        if not keypoint_path.is_file():
            raise DataFormatError(f"Manifest row {row_number}: CSV not found: {keypoint_path}")
        raw_index = (row.get("annotation_index") or "").strip()
        annotation_index = int(raw_index) if raw_index else None
        if annotation_index is not None and annotation_index < 0:
            raise DataFormatError(f"Manifest row {row_number}: annotation_index must be non-negative")
        pairs.append(
            SourcePair(
                annotation_json=annotation_path,
                keypoints_csv=keypoint_path,
                pairing_method="manifest",
                annotation_index=annotation_index,
                sample_id=(row.get("sample_id") or "").strip() or None,
            )
        )
    return pairs


def _stem_variants(value: str) -> set[str]:
    stem = unicodedata.normalize("NFC", Path(value).stem).casefold()
    simplified = _GENERIC_STEM_PARTS.sub("", stem)
    compact = re.sub(r"[^0-9a-z가-힣]+", "", simplified)
    variants = {compact} if compact else set()
    # Dataset videos such as Motion2-2 end in camera/channel 1..8. Preserve
    # the full form and add a channel-free form without stripping sample IDs.
    channel_stripped = re.sub(
        r"(?i)(?:(?<=motion\d)[-_][1-8]|[-_](?:cam(?:era)?|view|channel|ch)[-_]?[1-8])$",
        "",
        simplified,
    )
    channel_compact = re.sub(r"[^0-9a-z가-힣]+", "", channel_stripped)
    if channel_compact:
        variants.add(channel_compact)
    return variants


def _metadata_keys(path: Path, metadata: Mapping[str, Any]) -> set[str]:
    keys = _stem_variants(path.name)
    for field in ("video_name", "video_path"):
        value = metadata.get(field)
        if value:
            keys.update(_stem_variants(str(value)))
    return keys


def _csv_keys(path: Path) -> set[str]:
    return _stem_variants(path.name)


def _pair_score(
    json_path: Path,
    metadata_keys: set[str],
    csv_path: Path,
    csv_keys: set[str],
) -> float:
    if metadata_keys & csv_keys:
        score = 100.0
    else:
        score = max(
            (
                70.0 * SequenceMatcher(None, left, right).ratio()
                for left in metadata_keys
                for right in csv_keys
            ),
            default=0.0,
        )
    if json_path.parent == csv_path.parent:
        score += 20.0
    else:
        json_parents = {part.casefold() for part in json_path.parent.parts[-3:]}
        csv_parents = {part.casefold() for part in csv_path.parent.parts[-3:]}
        score += 5.0 * len(json_parents & csv_parents)
    return score


def discover_source_pairs(input_root: Path) -> DiscoveryResult:
    """Conservatively auto-pair matching files and report every ambiguity."""

    root = input_root.resolve()
    result = DiscoveryResult()
    csv_paths: list[Path] = []
    for path in sorted(item for item in root.rglob("*") if item.suffix.casefold() == ".csv"):
        if is_3d_keypoint_csv(path):
            csv_paths.append(path.resolve())
    result.three_dimensional_csv_count = len(csv_paths)
    keyed_csv = [(path, _csv_keys(path)) for path in csv_paths]

    for json_path in sorted(item for item in root.rglob("*") if item.suffix.casefold() == ".json"):
        resolved_json = json_path.resolve()
        try:
            metadata = load_metadata(resolved_json)
        except (OSError, DataFormatError) as error:
            safe_error = str(error).replace(str(root), "<input>").replace(root.as_posix(), "<input>")
            result.unreadable_files.append(
                {"path": str(json_path.relative_to(root)), "error": safe_error}
            )
            continue
        if not any(iter_normal_air_squat_annotations(metadata)):
            continue
        result.normal_air_squat_json_count += 1
        keys = _metadata_keys(resolved_json, metadata)
        scored = [
            (_pair_score(resolved_json, keys, csv_path, csv_keys), csv_path)
            for csv_path, csv_keys in keyed_csv
        ]
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        relative_json = str(resolved_json.relative_to(root))
        if not scored or scored[0][0] < 80.0:
            result.unmatched_json.append(relative_json)
            continue
        if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-9:
            result.ambiguous_json.append(relative_json)
            continue
        result.pairs.append(
            SourcePair(
                annotation_json=resolved_json,
                keypoints_csv=scored[0][1],
                pairing_method="automatic",
            )
        )
    return result
