"""Tests for AI Hub byte-offset ``.part`` TAR archives.

The real ``body_01.tar`` archive is tens of gigabytes.  These tests create a
small TAR in memory and split it using the same ``.part<offset>`` convention,
so no AI Hub records are copied into the repository.
"""

from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_trainer.reference.builder import BuildConfig, BuildError, build_reference
from ai_trainer.reference.io import discover_source_pairs
from ai_trainer.reference.parts import (
    PartSetError,
    discover_part_set,
    extract_tar,
    list_tar_members,
    merge_parts,
)


def make_tar(members: dict[str, bytes]) -> bytes:
    """Return a deterministic, uncompressed POSIX TAR containing *members*."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def write_parts(
    root: Path,
    payload: bytes,
    cuts: tuple[int, ...],
    *,
    archive_name: str = "body_01.tar",
) -> list[Path]:
    """Write parts whose suffix is the byte offset in the merged archive."""

    boundaries = (*cuts, len(payload))
    paths: list[Path] = []
    for start, end in zip(boundaries, boundaries[1:]):
        path = root / f"{archive_name}.part{start}"
        path.write_bytes(payload[start:end])
        paths.append(path)
    return paths


class PartDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tar_bytes = make_tar(
            {
                "Day05/subject01/camera01/frame_000001.jpg": b"\xff\xd8\xffsynthetic-1\xff\xd9",
                "Day05/subject01/camera01/frame_000002.jpg": b"\xff\xd8\xffsynthetic-2\xff\xd9",
            }
        )

    def test_numeric_suffix_is_a_byte_offset_not_a_part_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Lexical order is part0, part1400, part700; correct order is numeric.
            write_parts(root, self.tar_bytes, (0, 700, 1400))

            part_set = discover_part_set(root)

            self.assertEqual([part.offset for part in part_set.parts], [0, 700, 1400])
            self.assertEqual(
                [part.size for part in part_set.parts],
                [700, 700, len(self.tar_bytes) - 1400],
            )
            self.assertEqual(part_set.total_size, len(self.tar_bytes))
            self.assertEqual(part_set.archive_name, "body_01.tar")

    def test_archive_name_selects_one_set_when_directory_has_several(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_parts(root, self.tar_bytes, (0, 700), archive_name="body_01.tar")
            write_parts(root, self.tar_bytes, (0, 900), archive_name="body_02.tar")

            selected = discover_part_set(root, archive_name="body_02.tar")

            self.assertEqual(selected.archive_name, "body_02.tar")
            with self.assertRaisesRegex(
                PartSetError, "(?i)multiple|archive_name|more than one"
            ):
                discover_part_set(root)

    def test_gap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "body_01.tar.part0").write_bytes(self.tar_bytes[:700])
            (root / "body_01.tar.part701").write_bytes(self.tar_bytes[700:])

            with self.assertRaisesRegex(
                PartSetError, "(?i)gap|offset 701|expected 700"
            ):
                discover_part_set(root)

    def test_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "body_01.tar.part0").write_bytes(self.tar_bytes[:700])
            (root / "body_01.tar.part699").write_bytes(self.tar_bytes[700:])

            with self.assertRaisesRegex(
                PartSetError, "(?i)overlap|offset 699|expected 700"
            ):
                discover_part_set(root)

    def test_empty_part_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "body_01.tar.part0").write_bytes(self.tar_bytes)
            (root / f"body_01.tar.part{len(self.tar_bytes)}").write_bytes(b"")

            with self.assertRaisesRegex(PartSetError, "(?i)empty|zero"):
                discover_part_set(root)

    def test_missing_first_offset_and_invalid_tar_bookends_are_rejected(self) -> None:
        cases = {
            "missing offset zero": [(1, self.tar_bytes)],
            "invalid tar header": [
                (0, self.tar_bytes[:257] + b"xxxxx" + self.tar_bytes[262:])
            ],
            "invalid tar footer": [(0, self.tar_bytes[:-1024] + (b"X" * 1024))],
        }
        for label, parts in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for offset, payload in parts:
                    (root / f"body_01.tar.part{offset}").write_bytes(payload)
                with self.assertRaises(PartSetError):
                    discover_part_set(root)


class PartProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.member_payloads = {
            "Day05/subject01/camera01/frame_000001.jpg": b"\xff\xd8\xffsynthetic-1\xff\xd9",
            "Day05/subject01/camera01/frame_000002.jpg": b"\xff\xd8\xffsynthetic-2\xff\xd9",
            "README.txt": b"synthetic archive",
        }
        self.tar_bytes = make_tar(self.member_payloads)

    def _part_set(self, root: Path):
        write_parts(root, self.tar_bytes, (0, 700, 1400, 4097))
        return discover_part_set(root)

    def test_members_are_listed_without_materializing_a_merged_tar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            part_set = self._part_set(root)

            names = list_tar_members(part_set)
            limited = list_tar_members(part_set, limit=1)

            self.assertEqual(names, list(self.member_payloads))
            self.assertEqual(limited, [next(iter(self.member_payloads))])
            self.assertFalse((root / "body_01.tar").exists())

    def test_merge_streams_exact_bytes_and_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            part_set = self._part_set(root)
            output = root / "merged" / "body_01.tar"

            result = merge_parts(part_set, output)

            self.assertEqual(result, output.resolve())
            self.assertEqual(output.read_bytes(), self.tar_bytes)
            with self.assertRaisesRegex(PartSetError, "(?i)exists|overwrite"):
                merge_parts(part_set, output)
            self.assertEqual(merge_parts(part_set, output, overwrite=True), output.resolve())
            self.assertEqual(output.read_bytes(), self.tar_bytes)

    def test_selective_extract_and_jpg_only_data_cannot_build_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            part_set = self._part_set(root)
            extracted = root / "extracted"
            selected_name = "Day05/subject01/camera01/frame_000002.jpg"

            result = extract_tar(part_set, extracted, members=[selected_name])

            self.assertEqual(result, extracted.resolve())
            self.assertEqual((extracted / selected_name).read_bytes(), self.member_payloads[selected_name])
            self.assertFalse(
                (extracted / "Day05/subject01/camera01/frame_000001.jpg").exists()
            )

            discovery = discover_source_pairs(extracted)
            self.assertEqual(discovery.normal_air_squat_json_count, 0)
            self.assertEqual(discovery.three_dimensional_csv_count, 0)
            self.assertEqual(discovery.pairs, [])
            with self.assertRaisesRegex(BuildError, "JSON/3-D CSV"):
                build_reference(
                    BuildConfig(input_root=extracted, output_dir=root / "reference")
                )

    def test_extract_rejects_path_traversal_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            malicious = make_tar({"../outside.txt": b"must not escape"})
            write_parts(root, malicious, (0, 700))
            part_set = discover_part_set(root)

            with self.assertRaisesRegex(
                PartSetError, "(?i)unsafe|outside|traversal"
            ):
                extract_tar(part_set, root / "extracted")
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
