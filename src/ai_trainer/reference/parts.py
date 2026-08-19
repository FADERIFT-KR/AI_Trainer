"""Validate and process AI Hub ``.part<offset>`` archive fragments.

AI Hub names each fragment with its byte offset in the original archive, for
example ``body_01.tar.part1073741824``.  Lexicographic filename order is not
safe for these files; every operation in this module validates and uses the
numeric offset instead.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


_PART_PATTERN = re.compile(r"^(?P<archive>.+)\.part(?P<offset>\d+)$", re.IGNORECASE)
_COPY_BUFFER_SIZE = 8 * 1024 * 1024


class PartSetError(ValueError):
    """Raised when split files are incomplete, ambiguous, or unsafe."""


@dataclass(frozen=True)
class PartInfo:
    """One byte range in a split archive."""

    path: Path
    offset: int
    size: int


@dataclass(frozen=True)
class PartSet:
    """A validated, contiguous set of archive fragments."""

    archive_name: str
    parts: tuple[PartInfo, ...]
    total_size: int

    @property
    def directory(self) -> Path:
        return self.parts[0].path.parent


def _read_tail(parts: tuple[PartInfo, ...], size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    for part in reversed(parts):
        if remaining <= 0:
            break
        count = min(remaining, part.size)
        with part.path.open("rb") as handle:
            handle.seek(part.size - count)
            chunks.append(handle.read(count))
        remaining -= count
    return b"".join(reversed(chunks))


def _validate_archive_markers(part_set: PartSet) -> None:
    archive_name = part_set.archive_name.casefold()
    with part_set.parts[0].path.open("rb") as handle:
        header = handle.read(512)

    if archive_name.endswith(".tar"):
        if len(header) != 512 or header[257:262] != b"ustar":
            raise PartSetError(
                f"The first fragment is not a POSIX TAR header: "
                f"{part_set.parts[0].path}"
            )
        tail = _read_tail(part_set.parts, 1024)
        if len(tail) != 1024 or any(tail):
            raise PartSetError(
                "The split TAR has no two-block zero terminator; the final "
                "fragment may be incomplete."
            )
    elif archive_name.endswith(".zip"):
        if header[:4] not in {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}:
            raise PartSetError(
                f"The first fragment is not a ZIP header: {part_set.parts[0].path}"
            )


def discover_part_set(
    path_or_dir: str | Path,
    archive_name: str | None = None,
) -> PartSet:
    """Discover one archive and verify offset continuity and format markers.

    ``path_or_dir`` may be a directory containing fragments or any one fragment.
    When a directory contains multiple archives, ``archive_name`` is required.
    """

    source = Path(path_or_dir).expanduser()
    if source.is_file():
        match = _PART_PATTERN.match(source.name)
        if match is None:
            raise PartSetError(f"Not an AI Hub .part<offset> file: {source}")
        directory = source.parent
        inferred_name = match.group("archive")
        if archive_name is not None and Path(archive_name).name.casefold() != inferred_name.casefold():
            raise PartSetError(
                f"Fragment belongs to {inferred_name!r}, not {archive_name!r}."
            )
        archive_name = inferred_name
    elif source.is_dir():
        directory = source
    else:
        raise PartSetError(f"Part directory or fragment not found: {source}")

    groups: dict[str, list[tuple[Path, int]]] = {}
    canonical_names: dict[str, str] = {}
    for candidate in directory.iterdir():
        if not candidate.is_file():
            continue
        match = _PART_PATTERN.match(candidate.name)
        if match is None:
            continue
        base_name = match.group("archive")
        key = base_name.casefold()
        canonical_names.setdefault(key, base_name)
        groups.setdefault(key, []).append((candidate.resolve(), int(match.group("offset"))))

    if archive_name is not None:
        requested = Path(archive_name).name.casefold()
        if requested not in groups:
            available = ", ".join(sorted(canonical_names.values())) or "none"
            raise PartSetError(
                f"No fragments for {archive_name!r} in {directory}. Available: {available}"
            )
        selected_key = requested
    else:
        if not groups:
            raise PartSetError(f"No .part<offset> files found in: {directory}")
        if len(groups) != 1:
            available = ", ".join(sorted(canonical_names.values()))
            raise PartSetError(
                f"Multiple split archives found ({available}); specify archive_name."
            )
        selected_key = next(iter(groups))

    candidates = sorted(groups[selected_key], key=lambda item: item[1])
    parts: list[PartInfo] = []
    expected_offset = 0
    for path, offset in candidates:
        size = path.stat().st_size
        if size <= 0:
            raise PartSetError(f"Empty archive fragment: {path}")
        if offset != expected_offset:
            relation = "gap" if offset > expected_offset else "overlap"
            raise PartSetError(
                f"Archive fragment {relation}: expected offset {expected_offset}, "
                f"found {offset} in {path.name}."
            )
        parts.append(PartInfo(path=path, offset=offset, size=size))
        expected_offset += size

    result = PartSet(
        archive_name=canonical_names[selected_key],
        parts=tuple(parts),
        total_size=expected_offset,
    )
    _validate_archive_markers(result)
    return result


class _ConcatenatedPartStream(io.RawIOBase):
    """Forward-only binary stream over a validated part set."""

    def __init__(self, part_set: PartSet) -> None:
        super().__init__()
        self._parts = part_set.parts
        self._part_index = 0
        self._handle: io.BufferedReader | None = None
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._position

    def _open_current(self) -> bool:
        if self._part_index >= len(self._parts):
            return False
        if self._handle is None:
            self._handle = self._parts[self._part_index].path.open("rb")
        return True

    def readinto(self, buffer: object) -> int:
        view = memoryview(buffer).cast("B")
        total = 0
        while total < len(view) and self._open_current():
            assert self._handle is not None
            count = self._handle.readinto(view[total:])
            if count:
                total += count
                self._position += count
                continue
            self._handle.close()
            self._handle = None
            self._part_index += 1
        return total

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def _open_split_tar(part_set: PartSet) -> tuple[tarfile.TarFile, io.BufferedReader]:
    if not part_set.archive_name.casefold().endswith(".tar"):
        raise PartSetError(
            f"Streaming extraction currently supports .tar parts, not {part_set.archive_name!r}. "
            "Use merge_parts first for this archive type."
        )
    buffered = io.BufferedReader(_ConcatenatedPartStream(part_set), _COPY_BUFFER_SIZE)
    try:
        archive = tarfile.open(fileobj=buffered, mode="r|")
    except Exception:
        buffered.close()
        raise
    return archive, buffered


def merge_parts(
    part_set: PartSet,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Join fragments into one archive using an atomic destination replace."""

    destination = Path(output).expanduser().resolve()
    if destination in {part.path for part in part_set.parts}:
        raise PartSetError("Merged output cannot overwrite one of its source fragments.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise PartSetError(f"Merged archive already exists: {destination}")
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < part_set.total_size:
        raise PartSetError(
            f"Not enough free space to merge {part_set.total_size:,} bytes; "
            f"only {free_bytes:,} bytes are available at {destination.parent}."
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle:
            temporary_path = Path(output_handle.name)
            for part in part_set.parts:
                with part.path.open("rb") as input_handle:
                    shutil.copyfileobj(input_handle, output_handle, _COPY_BUFFER_SIZE)
            output_handle.flush()
            os.fsync(output_handle.fileno())

        if temporary_path.stat().st_size != part_set.total_size:
            raise PartSetError(
                "Merged archive size does not match the validated logical size."
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def list_tar_members(part_set: PartSet, limit: int | None = None) -> list[str]:
    """List member names directly from split TAR data without creating a TAR file."""

    if limit is not None and limit < 0:
        raise PartSetError("Member list limit cannot be negative.")
    if limit == 0:
        return []

    names: list[str] = []
    archive, buffered = _open_split_tar(part_set)
    try:
        for member in archive:
            names.append(member.name)
            if limit is not None and len(names) >= limit:
                break
    finally:
        archive.close()
        buffered.close()
    return names


def _safe_target(destination: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    pure_path = PurePosixPath(normalized)
    if pure_path.is_absolute() or re.match(r"^[A-Za-z]:", normalized):
        raise PartSetError(f"Unsafe absolute TAR member path: {member_name!r}")
    components = tuple(part for part in pure_path.parts if part not in {"", "."})
    if not components or ".." in components:
        raise PartSetError(f"Unsafe TAR member path: {member_name!r}")
    target = destination.joinpath(*components).resolve()
    try:
        target.relative_to(destination)
    except ValueError as error:
        raise PartSetError(f"TAR member escapes destination: {member_name!r}") from error
    return target


def extract_tar(
    part_set: PartSet,
    destination: str | Path,
    *,
    overwrite: bool = False,
    members: Iterable[str] | None = None,
) -> Path:
    """Safely stream-extract regular files and directories from split TAR parts.

    No intermediate merged TAR is written.  Optional ``members`` are exact TAR
    member names, which is useful for selective extraction.
    """

    output_root = Path(destination).expanduser().resolve()
    if output_root.exists() and not output_root.is_dir():
        raise PartSetError(f"Extraction destination is not a directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(output_root.iterdir()):
        raise PartSetError(
            f"Extraction destination is not empty: {output_root}. "
            "Choose an empty directory or pass overwrite=True."
        )

    requested = None if members is None else {name.replace("\\", "/") for name in members}
    requested_remaining = None if requested is None else set(requested)
    archive, buffered = _open_split_tar(part_set)
    try:
        for member in archive:
            normalized_name = member.name.replace("\\", "/")
            if requested is not None and normalized_name not in requested:
                continue
            if requested_remaining is not None:
                requested_remaining.discard(normalized_name)
            target = _safe_target(output_root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise PartSetError(
                    f"Unsupported TAR member type for safe extraction: {member.name!r}"
                )
            if target.exists() and not overwrite:
                raise PartSetError(f"Extraction target already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source_handle = archive.extractfile(member)
            if source_handle is None:
                raise PartSetError(f"Cannot read TAR member: {member.name!r}")

            temporary_path: Path | None = None
            try:
                with source_handle, tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as output_handle:
                    temporary_path = Path(output_handle.name)
                    shutil.copyfileobj(source_handle, output_handle, _COPY_BUFFER_SIZE)
                os.replace(temporary_path, target)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        if requested_remaining:
            missing = ", ".join(sorted(requested_remaining))
            raise PartSetError(f"Requested TAR members were not found: {missing}")
    except (tarfile.TarError, OSError) as error:
        if isinstance(error, PartSetError):
            raise
        raise PartSetError(f"Split TAR extraction failed: {error}") from error
    finally:
        archive.close()
        buffered.close()
    return output_root


__all__ = [
    "PartInfo",
    "PartSet",
    "PartSetError",
    "discover_part_set",
    "extract_tar",
    "list_tar_members",
    "merge_parts",
]
