"""End-to-end AI Hub air-squat reference data builder."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io import (
    DataFormatError,
    DiscoveryResult,
    SourcePair,
    discover_source_pairs,
    iter_normal_air_squat_annotations,
    load_metadata,
    read_3d_keypoints,
    read_pair_manifest,
    slice_annotation,
)
from .processing import (
    ProcessedRepetition,
    ProcessingError,
    aggregate_repetitions,
    process_sequence,
)
from .schema import (
    COMMON_JOINTS,
    DATASET_ID,
    DATASET_NAME,
    DATASET_URL,
    DATASET_VERSION,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    USAGE_POLICY_URL,
    schema_metadata,
)
from .visualization import render_reference_preview


class BuildError(RuntimeError):
    """Raised when reference data cannot be built safely."""


@dataclass(frozen=True)
class BuildConfig:
    input_root: Path
    output_dir: Path
    pairs_manifest: Path | None = None
    target_frames: int = 101
    fps: float = 30.0
    min_frames: int = 15
    min_flexion_deg: float = 25.0
    min_valid_ratio: float = 0.95
    max_missing_gap_frames: int = 2
    min_repetitions: int = 3
    outlier_mad_threshold: float = 4.0
    include_repetitions: bool = True
    include_plot: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    discovered_pairs: int
    processed_annotations: int
    total_repetitions: int
    accepted_repetitions: int
    rejected_repetitions: int
    skipped_sources: int
    preview_path: Path | None


@dataclass
class _RepetitionRecord:
    repetition: ProcessedRepetition
    row: dict[str, Any]


_OUTPUT_FILES = (
    "reference.npz",
    "repetitions.npz",
    "manifest.csv",
    "metadata.json",
    "build_report.json",
    "reference_preview.png",
)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: Any) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _atomic_write(path: Path, writer: Callable[[Any], None], *, binary: bool) -> None:
    mode = "wb" if binary else "w"
    kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False, **kwargs) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    assert temporary is not None
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        lambda handle: json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True),
        binary=False,
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    _atomic_write(path, lambda handle: np.savez_compressed(handle, **arrays), binary=True)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])

    def write(handle: Any) -> None:
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    _atomic_write(path, write, binary=False)


def _load_pairs(config: BuildConfig) -> tuple[list[SourcePair], DiscoveryResult | None]:
    if config.pairs_manifest is not None:
        manifest = config.pairs_manifest.resolve()
        if not manifest.is_file():
            raise BuildError(f"Pair manifest not found: {manifest}")
        return read_pair_manifest(manifest, config.input_root), None
    discovery = discover_source_pairs(config.input_root)
    return discovery.pairs, discovery


def _validate_config(config: BuildConfig) -> None:
    if not config.input_root.resolve().is_dir():
        raise BuildError(f"Input directory not found: {config.input_root}")
    if config.target_frames < 5 or config.target_frames % 2 == 0:
        raise BuildError("target_frames must be an odd number of at least 5")
    if config.fps <= 0:
        raise BuildError("fps must be positive")
    if config.min_repetitions < 1:
        raise BuildError("min_repetitions must be at least 1")
    if config.min_frames < 5:
        raise BuildError("min_frames must be at least 5")
    if not 0.0 < config.min_valid_ratio <= 1.0:
        raise BuildError("min_valid_ratio must be in (0, 1]")
    if config.max_missing_gap_frames < 0:
        raise BuildError("max_missing_gap_frames must be non-negative")


def _prepare_output(config: BuildConfig) -> Path:
    output = config.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in _OUTPUT_FILES if (output / name).exists()]
    if existing and not config.overwrite:
        names = ", ".join(path.name for path in existing)
        raise BuildError(f"Output already exists ({names}); pass --overwrite to replace it")
    return output


def _discovery_json(discovery: DiscoveryResult | None) -> dict[str, Any]:
    if discovery is None:
        return {"mode": "manifest"}
    return {
        "mode": "automatic",
        "normal_air_squat_json_count": discovery.normal_air_squat_json_count,
        "three_dimensional_csv_count": discovery.three_dimensional_csv_count,
        "unmatched_json": discovery.unmatched_json,
        "ambiguous_json": discovery.ambiguous_json,
        "unreadable_files": discovery.unreadable_files,
    }


def _safe_error(error: Exception, input_root: Path) -> str:
    message = str(error)
    roots = {str(input_root.resolve()), input_root.resolve().as_posix()}
    for root in roots:
        message = message.replace(root, "<input>")
    return message


def build_reference(config: BuildConfig) -> BuildResult:
    """Build local reference artifacts from approved/downloaded AI Hub files."""

    config = BuildConfig(
        **{
            **asdict(config),
            "input_root": Path(config.input_root).resolve(),
            "output_dir": Path(config.output_dir).resolve(),
            "pairs_manifest": (
                Path(config.pairs_manifest).resolve() if config.pairs_manifest is not None else None
            ),
        }
    )
    _validate_config(config)
    pairs, discovery = _load_pairs(config)
    if not pairs:
        detail = ""
        if discovery is not None:
            detail = (
                f" Found {discovery.normal_air_squat_json_count} matching JSON and "
                f"{discovery.three_dimensional_csv_count} 3-D CSV files; "
                f"{len(discovery.unmatched_json)} unmatched and "
                f"{len(discovery.ambiguous_json)} ambiguous."
            )
        raise BuildError(
            "No unambiguous normal air-squat JSON/3-D CSV pairs were found."
            f"{detail} Provide --pairs-manifest for the archive's exact mapping."
        )

    records: list[_RepetitionRecord] = []
    skipped: list[dict[str, str]] = []
    processed_annotations = 0
    seen_segments: set[tuple[str, int, int]] = set()
    hash_cache: dict[Path, str] = {}

    for pair in pairs:
        relative_json = _relative(pair.annotation_json, config.input_root)
        relative_csv = _relative(pair.keypoints_csv, config.input_root)
        try:
            metadata = load_metadata(pair.annotation_json)
            sequence = read_3d_keypoints(pair.keypoints_csv)
        except (OSError, DataFormatError) as error:
            skipped.append(
                {
                    "source": f"{relative_json} | {relative_csv}",
                    "error": _safe_error(error, config.input_root),
                }
            )
            continue

        selected_annotations = list(
            iter_normal_air_squat_annotations(metadata, pair.annotation_index)
        )
        if not selected_annotations:
            skipped.append(
                {
                    "source": relative_json,
                    "error": "No normal air-squat annotation matched the requested index",
                }
            )
            continue

        for annotation_index, annotation in selected_annotations:
            try:
                sliced = slice_annotation(sequence, annotation, fps=config.fps)
                dedupe_key = (str(pair.keypoints_csv.resolve()), sliced.row_start, sliced.row_end)
                if dedupe_key in seen_segments:
                    continue
                seen_segments.add(dedupe_key)
                repetitions = process_sequence(
                    sliced.points,
                    target_frames=config.target_frames,
                    min_flexion_deg=config.min_flexion_deg,
                    min_frames=config.min_frames,
                    min_valid_ratio=config.min_valid_ratio,
                    max_missing_gap_frames=config.max_missing_gap_frames,
                )
            except (DataFormatError, ProcessingError) as error:
                skipped.append(
                    {
                        "source": f"{relative_json}#{annotation_index} | {relative_csv}",
                        "error": _safe_error(error, config.input_root),
                    }
                )
                continue

            processed_annotations += 1
            if pair.annotation_json not in hash_cache:
                hash_cache[pair.annotation_json] = _sha256(pair.annotation_json)
            if pair.keypoints_csv not in hash_cache:
                hash_cache[pair.keypoints_csv] = _sha256(pair.keypoints_csv)

            for repetition_index, repetition in enumerate(repetitions):
                sample_id = pair.sample_id or _stable_id(
                    relative_json,
                    relative_csv,
                    annotation_index,
                    sliced.row_start,
                    sliced.row_end,
                )
                repetition_id = (
                    f"{sample_id}-a{annotation_index:03d}-r{repetition_index + 1:02d}"
                )
                source_start = sliced.row_start + repetition.source_start
                source_bottom = sliced.row_start + repetition.source_bottom
                source_end = sliced.row_start + repetition.source_end
                row = {
                    "repetition_id": repetition_id,
                    "source_annotation_json": relative_json,
                    "source_keypoints_3d_csv": relative_csv,
                    "annotation_index": annotation_index,
                    "annotation_no": annotation.get("annotation_no", ""),
                    "pairing_method": pair.pairing_method,
                    "source_row_start": source_start,
                    "source_row_bottom": source_bottom,
                    "source_row_end": source_end,
                    "source_frame_start": (
                        ""
                        if sliced.source_frame_start is None
                        else sliced.source_frame_start + repetition.source_start
                    ),
                    "source_frame_end": (
                        ""
                        if sliced.source_frame_start is None
                        else sliced.source_frame_start + repetition.source_end
                    ),
                    "target_frames": config.target_frames,
                    "bottom_phase": round(repetition.bottom_phase, 8),
                    "flexion_range_deg": round(repetition.flexion_range_deg, 6),
                    "valid_joint_ratio": round(repetition.valid_ratio, 8),
                    "annotation_sha256": hash_cache[pair.annotation_json],
                    "keypoints_sha256": hash_cache[pair.keypoints_csv],
                    "accepted": "",
                    "template_distance": "",
                }
                records.append(_RepetitionRecord(repetition=repetition, row=row))

    if len(records) < config.min_repetitions:
        raise BuildError(
            f"Only {len(records)} complete repetitions were produced; "
            f"at least {config.min_repetitions} are required. "
            f"Skipped source intervals: {len(skipped)}."
        )

    repetitions = [record.repetition for record in records]
    arrays, keep_mask, distances = aggregate_repetitions(
        repetitions,
        mad_threshold=config.outlier_mad_threshold,
    )
    accepted_count = int(keep_mask.sum())
    if accepted_count < config.min_repetitions:
        raise BuildError(
            f"Only {accepted_count} repetitions remain after outlier filtering; "
            f"at least {config.min_repetitions} are required"
        )
    for record, accepted, distance in zip(records, keep_mask, distances):
        record.row["accepted"] = "true" if accepted else "false"
        record.row["template_distance"] = round(float(distance), 8)

    output = _prepare_output(config)
    repetition_ids = np.asarray([record.row["repetition_id"] for record in records])
    reference_arrays = {
        **arrays,
        "joint_names": np.asarray(COMMON_JOINTS),
        "feature_names": np.asarray(FEATURE_NAMES),
        "phase_names": np.asarray(("descent", "bottom", "ascent")),
        "accepted_repetition_count": np.asarray(accepted_count, dtype=np.int32),
        "total_repetition_count": np.asarray(len(records), dtype=np.int32),
    }
    _write_npz(output / "reference.npz", reference_arrays)

    if config.include_repetitions:
        repetition_arrays = {
            "positions": np.stack([record.repetition.positions for record in records]),
            "features": np.stack([record.repetition.features for record in records]),
            "repetition_ids": repetition_ids,
            "accepted": keep_mask.astype(np.bool_),
            "template_distance": distances,
            "bottom_phase": np.asarray(
                [record.repetition.bottom_phase for record in records], dtype=np.float32
            ),
            "joint_names": np.asarray(COMMON_JOINTS),
            "feature_names": np.asarray(FEATURE_NAMES),
        }
        _write_npz(output / "repetitions.npz", repetition_arrays)
    elif (output / "repetitions.npz").exists() and config.overwrite:
        # Avoid leaving a stale optional artifact when rebuilding without it.
        (output / "repetitions.npz").unlink()

    _write_manifest(output / "manifest.csv", [record.row for record in records])
    config_metadata = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
        if key not in {"input_root", "output_dir", "pairs_manifest", "overwrite"}
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_id": DATASET_ID,
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "dataset_url": DATASET_URL,
            "usage_policy_url": USAGE_POLICY_URL,
            "attribution": (
                "과학기술정보통신부·한국지능정보사회진흥원 AI Hub "
                f"'{DATASET_NAME}'(데이터셋 {DATASET_ID}) 활용"
            ),
            "redistribution_notice": (
                "AI Hub 이용정책에 따라 원본 및 파생 데이터의 열람·제공·양도 제한을 확인할 것"
            ),
        },
        "selection": {
            "motion": "에어 스쿼트",
            "correctness": "정상",
            "processed_annotations": processed_annotations,
            "total_repetitions": len(records),
            "accepted_repetitions": accepted_count,
            "rejected_repetitions": int((~keep_mask).sum()),
            "subject_balanced": False,
            "subject_balance_note": (
                "공개 JSON 스키마에는 신뢰 가능한 actor ID가 없어 피험자별 가중치를 적용하지 않음"
            ),
        },
        "config": config_metadata,
        "schema": schema_metadata(),
    }
    _write_json(output / "metadata.json", metadata)
    _write_json(
        output / "build_report.json",
        {
            "discovery": _discovery_json(discovery),
            "discovered_pairs": len(pairs),
            "processed_annotations": processed_annotations,
            "skipped_sources": skipped,
        },
    )

    # The preview is deliberately the final build artifact: it visualizes the
    # exact aggregate arrays and metadata that were successfully persisted.
    preview_path: Path | None = None
    if config.include_plot:
        preview_path = output / "reference_preview.png"
        _atomic_write(
            preview_path,
            lambda handle: render_reference_preview(
                arrays["positions_median"],
                COMMON_JOINTS,
                int(np.asarray(arrays["bottom_index"]).item()),
                handle,
            ),
            binary=True,
        )
    elif (output / "reference_preview.png").exists() and config.overwrite:
        # Avoid keeping a preview that does not belong to this rebuild.
        (output / "reference_preview.png").unlink()

    return BuildResult(
        output_dir=output,
        discovered_pairs=len(pairs),
        processed_annotations=processed_annotations,
        total_repetitions=len(records),
        accepted_repetitions=accepted_count,
        rejected_repetitions=int((~keep_mask).sum()),
        skipped_sources=len(skipped),
        preview_path=preview_path,
    )
