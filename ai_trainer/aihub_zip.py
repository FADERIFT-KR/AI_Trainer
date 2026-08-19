"""AI Hub 크로스핏(에어스쿼트) 라벨링 데이터(TL.zip / VL.zip) 접근 유틸리티.

zip 파일을 압축 해제하지 않고 ``zipfile``로 직접 열어 필요한 CSV/JSON만
메모리에서 읽는다. zip 내부 한글 파일명은 cp437로 잘못 인코딩되어 있으므로
cp949로 복원해서 사용한다.

디렉터리 규칙 (에어스쿼트 기준)::

    스쿼트/에어스쿼트/{오류유형}/{난이도}/{actorID}/{repIndex}/
        3d_points.csv
        camera{0..7}/local_keypoints/*.csv   (2D)
        camera{0..7}/video/annotation.json
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# 3d_points.csv / camera{N}/local_keypoints/*.csv 공통 26관절 순서
JOINT_NAMES = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
    "Head", "Neck", "Hip",
    "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
]

_SEQ_3D_RE = re.compile(
    r"^스쿼트/에어스쿼트/([^/]+)/([^/]+)/([^/]+)/(\d+)/3d_points\.csv$"
)


def decode_name(name: str) -> str:
    """zip 내부 한글 파일명(cp437로 깨진 상태)을 cp949로 복원한다."""
    try:
        return name.encode("cp437").decode("cp949")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


@dataclass(frozen=True)
class SequenceKey:
    """에어스쿼트 시퀀스 1개(= 특정 피험자의 스쿼트 1회 반복)를 식별한다."""

    error_type: str  # 정상 / 발뒤꿈치오류 / 엉덩이하방오류 / 고관절오류
    level: str        # 초급 / 중급 / 고급
    actor: str        # CAxx / CBxx / CIxx
    rep: str          # 반복 인덱스 (문자열, 예: "1")

    @property
    def base_dir(self) -> str:
        return f"스쿼트/에어스쿼트/{self.error_type}/{self.level}/{self.actor}/{self.rep}"

    def __str__(self) -> str:  # 사람이 읽기 좋은 표시용
        return f"{self.actor}/{self.error_type}/{self.level}/rep{self.rep}"


class AiHubZip:
    """TL.zip 또는 VL.zip 한 개를 감싸는 read-only 접근자."""

    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"zip 파일을 찾을 수 없습니다: {self.zip_path}")
        self._zf = zipfile.ZipFile(self.zip_path)
        self._name_map: dict[str, zipfile.ZipInfo] = {
            decode_name(zi.filename): zi for zi in self._zf.infolist()
        }

    def close(self) -> None:
        self._zf.close()

    def __enter__(self) -> "AiHubZip":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _read(self, decoded_path: str) -> bytes:
        try:
            zi = self._name_map[decoded_path]
        except KeyError as e:
            raise FileNotFoundError(f"zip 내부 경로를 찾을 수 없습니다: {decoded_path}") from e
        return self._zf.read(zi)

    def iter_air_squat_sequences(self) -> Iterator[SequenceKey]:
        """에어스쿼트 하위의 모든 (오류유형/난이도/actor/rep) 시퀀스를 나열한다."""
        for name in self._name_map:
            m = _SEQ_3D_RE.match(name)
            if m:
                yield SequenceKey(*m.groups())

    def find_sequences(
        self,
        error_type: str | None = None,
        level: str | None = None,
        actor: str | None = None,
        rep: str | None = None,
    ) -> list[SequenceKey]:
        """조건에 맞는 시퀀스를 정렬된 리스트로 반환한다. 조건은 모두 선택적."""
        out = []
        for seq in self.iter_air_squat_sequences():
            if error_type and seq.error_type != error_type:
                continue
            if level and seq.level != level:
                continue
            if actor and seq.actor != actor:
                continue
            if rep and seq.rep != rep:
                continue
            out.append(seq)
        return sorted(out, key=lambda s: (s.error_type, s.level, s.actor, int(s.rep)))

    def list_cameras(self, seq: SequenceKey) -> list[int]:
        """시퀀스에 존재하는 camera 번호 목록(0~7 중 실제 존재하는 것)을 반환한다."""
        cams = set()
        prefix = seq.base_dir + "/camera"
        for name in self._name_map:
            if name.startswith(prefix):
                m = re.match(rf"^{re.escape(prefix)}(\d+)/", name)
                if m:
                    cams.add(int(m.group(1)))
        return sorted(cams)

    def read_3d(self, seq: SequenceKey) -> tuple[list[str], np.ndarray]:
        """3d_points.csv를 (프레임 파일명 리스트, (T, 26, 3) ndarray)로 반환."""
        raw = self._read(f"{seq.base_dir}/3d_points.csv").decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(raw)))
        frames = [r["image_filename"] for r in rows]
        coords = np.zeros((len(rows), len(JOINT_NAMES), 3), dtype=np.float64)
        for t, r in enumerate(rows):
            for j, name in enumerate(JOINT_NAMES):
                coords[t, j] = (
                    float(r[f"{name}_x"]),
                    float(r[f"{name}_y"]),
                    float(r[f"{name}_z"]),
                )
        return frames, coords

    def read_2d(self, seq: SequenceKey, camera: int) -> tuple[list[str], np.ndarray]:
        """camera{N}/local_keypoints/*.csv를 (프레임 파일명 리스트, (T, 26, 2) ndarray)로 반환.

        실제 파일명이 항상 ``Motion2-N.csv``는 아니고(``Motion2-N - XofY.csv`` 형태도 존재)
        정확한 이름을 하드코딩하지 않고 디렉터리 내 csv를 탐색한다.
        """
        prefix = f"{seq.base_dir}/camera{camera}/local_keypoints/"
        candidates = sorted(
            n for n in self._name_map if n.startswith(prefix) and n.endswith(".csv")
        )
        if not candidates:
            raise FileNotFoundError(f"camera{camera} 2D CSV를 찾을 수 없습니다: {prefix}")
        raw = self._read(candidates[0]).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(raw)))
        frames = [r["image_filename"] for r in rows]
        coords = np.zeros((len(rows), len(JOINT_NAMES), 2), dtype=np.float64)
        for t, r in enumerate(rows):
            for j, name in enumerate(JOINT_NAMES):
                coords[t, j] = (float(r[f"{name}_x"]), float(r[f"{name}_y"]))
        return frames, coords

    def read_annotation(self, seq: SequenceKey, camera: int) -> dict:
        """camera{N}/video/annotation.json을 dict로 반환.

        start_frame/end_frame(스쿼트 동작 구간)과 actor 신체정보를 담고 있다.
        """
        path = f"{seq.base_dir}/camera{camera}/video/annotation.json"
        return json.loads(self._read(path).decode("utf-8-sig"))
