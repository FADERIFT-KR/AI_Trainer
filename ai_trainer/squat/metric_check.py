"""원천 영상(input) <-> 라벨링 3D 정답(output 대조용) 매칭과 좌표 매칭율 계산.

AI Hub 에어스쿼트 데이터는 원천(영상)과 라벨링(CSV)이 경로·파일명까지 대칭이라,
영상 하나를 고르면 대조할 CSV가 유일하게 결정된다.

    원천   {루트}/{클래스}/{난이도}/{배우}/{rep}/camera{N}/video/Motion2-{N+1} - {rep}of4.avi
    라벨   스쿼트/에어스쿼트/{클래스}/{난이도}/{배우}/{rep}/3d_points.csv          <- 3D 정답(26관절, mm)
                                             /{rep}/camera{N}/local_keypoints/Motion2-{N+1} - {rep}of4.csv

`cameraN` <-> `Motion2-(N+1)` 규칙과 시점 매핑(정면=camera1, 사선=camera2, 측면=camera7)은
`configs/view_condition_thresholds.json`의 실측 카메라 통계에서 확인했다.
"""
from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ai_trainer.core.s3_mapping.common_skeleton import (
    COMMON_FROM_AIHUB_IDX,
    COMMON_JOINT_NAMES,
)
from ai_trainer.squat.aihub_zip import JOINT_NAMES

LABEL_PREFIX = "스쿼트/에어스쿼트"
ERROR_TYPES = ("정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류")
LEVELS = ("초급", "중급", "고급")

# 8대 카메라 중 이 세 대가 정면/사선/측면이다(카메라 방위각 실측: +-10도 / +43도 / +79도).
VIEW_CAMERAS = {"front": 1, "oblique": 2, "side": 7}
VIEW_LABEL_KO = {"front": "정면", "oblique": "사선", "side": "측면"}

_HIP = COMMON_JOINT_NAMES.index("Hip")
_NECK = COMMON_JOINT_NAMES.index("Neck")
# 파일명 규칙이 클래스마다 다르다(실측): 정상은 "Motion2-2 - 1of4", 오류 클래스는
# "Motion2-2". 카메라 번호만 공통이고 rep은 폴더명에서 읽는다.
_VIDEO_NAME = re.compile(r"^Motion2-(\d+)(?: - (\d+)of(\d+))?$")


def nfc(text: str) -> str:
    """macOS 파일시스템은 한글 경로를 NFD(분해형)로 돌려준다.

    코드 안의 '정상' 같은 문자열은 NFC(조합형)이라 그대로 비교하면 눈에는 같아 보여도
    다른 문자열로 판정된다. 경로에서 읽은 이름은 전부 이 함수를 거쳐 비교·조립한다.
    """
    return unicodedata.normalize("NFC", text)


class MatchError(Exception):
    """영상 경로 해석이나 라벨 매칭이 실패했을 때 — 사용자에게 무엇이 문제인지 알린다."""

    def __init__(self, message: str, folder: Path | None = None):
        super().__init__(message)
        self.folder = folder


@dataclass(frozen=True)
class SourceClip:
    """원천 영상 한 개에서 읽어낸 식별 정보."""

    video_path: Path
    error_type: str
    level: str
    actor: str
    rep: str
    camera: int
    stem: str  # "Motion2-2 - 1of4"

    @property
    def view(self) -> str | None:
        for name, index in VIEW_CAMERAS.items():
            if index == self.camera:
                return name
        return None

    @property
    def label_dir(self) -> str:
        return f"{LABEL_PREFIX}/{self.error_type}/{self.level}/{self.actor}/{self.rep}"

    @property
    def ground_truth_3d(self) -> str:
        return f"{self.label_dir}/3d_points.csv"

    @property
    def keypoints_2d(self) -> str:
        return f"{self.label_dir}/camera{self.camera}/local_keypoints/{self.stem}.csv"

    def describe(self) -> str:
        view = VIEW_LABEL_KO.get(self.view or "", f"camera{self.camera}")
        return f"{self.error_type}/{self.level}/{self.actor}/rep{self.rep} · {view}"


def parse_source_video(video_path: str | Path) -> SourceClip:
    """원천 영상 경로에서 클래스/난이도/배우/rep/카메라를 읽는다.

    루트 위치는 상관하지 않는다 — 외장 하드의 `에어스쿼트/...`든, 배우 한 명만 복사해
    온 폴더든, `{...}/{rep}/camera{N}/video/{파일}` 꼴만 지켜지면 된다.
    """
    path = Path(video_path)
    parts = path.parts
    if len(parts) < 6 or path.parent.name != "video":
        raise MatchError(
            f"영상 경로가 AI Hub 원천데이터 구조가 아닙니다.\n"
            f"기대: .../{{배우}}/{{rep}}/camera{{N}}/video/{path.name}\n"
            f"실제: {path}",
            path.parent,
        )
    camera_dir = parts[-3]
    if not camera_dir.startswith("camera") or not camera_dir[6:].isdigit():
        raise MatchError(
            f"카메라 폴더 이름을 읽을 수 없습니다: '{camera_dir}' (camera0~camera7이어야 합니다)",
            path.parent,
        )
    camera = int(camera_dir[6:])
    rep, actor = nfc(parts[-4]), nfc(parts[-5])
    level = nfc(parts[-6])
    error_type = nfc(parts[-7]) if len(parts) >= 7 else ""

    match = _VIDEO_NAME.match(nfc(path.stem))
    if match is None:
        raise MatchError(
            f"영상 파일명이 규칙과 다릅니다: '{path.name}'\n"
            f"기대 형식: 'Motion2-{camera + 1} - {rep}of4.avi' 또는 'Motion2-{camera + 1}.avi'",
            path.parent,
        )
    motion_index, name_rep, _total = match.groups()
    if int(motion_index) != camera + 1:
        raise MatchError(
            f"영상 파일명과 카메라 폴더가 어긋납니다.\n"
            f"'{camera_dir}' 폴더라면 파일명은 Motion2-{camera + 1}이어야 하는데 "
            f"Motion2-{motion_index} 입니다: {path.name}",
            path.parent,
        )
    if name_rep is not None and name_rep != rep:
        raise MatchError(
            f"영상 파일명의 rep과 폴더가 어긋납니다.\n"
            f"폴더는 '{rep}'인데 파일명은 '{name_rep}of...' 입니다: {path.name}",
            path.parent,
        )
    if error_type not in ERROR_TYPES:
        raise MatchError(
            f"오류유형 폴더를 찾지 못했습니다: '{error_type}'\n"
            f"기대: {', '.join(ERROR_TYPES)} 중 하나",
            path.parent,
        )
    if level not in LEVELS:
        raise MatchError(
            f"난이도 폴더를 찾지 못했습니다: '{level}' (기대: {', '.join(LEVELS)})",
            path.parent,
        )
    return SourceClip(path, error_type, level, actor, rep, camera, nfc(path.stem))


def find_view_video(rep_dir: Path, camera: int, rep: str) -> Path | None:
    """`{rep_dir}/camera{N}/video/` 에서 두 파일명 규칙 중 있는 것을 찾는다."""
    video_dir = rep_dir / f"camera{camera}" / "video"
    for name in (f"Motion2-{camera + 1} - {rep}of4.avi", f"Motion2-{camera + 1}.avi"):
        candidate = video_dir / name
        if candidate.is_file():
            return candidate
    if video_dir.is_dir():  # 규칙 밖 이름이라도 그 카메라에 영상이 하나뿐이면 그것을 쓴다
        movies = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in (".avi", ".mp4"))
        if len(movies) == 1:
            return movies[0]
    return None


def sibling_view_videos(clip: SourceClip) -> dict[str, Path]:
    """같은 rep의 정면/사선/측면 영상 경로를 모두 돌려준다(없는 것은 제외)."""
    rep_dir = clip.video_path.parents[2]
    found: dict[str, Path] = {}
    for view, camera in VIEW_CAMERAS.items():
        candidate = find_view_video(rep_dir, camera, clip.rep)
        if candidate is not None:
            found[view] = candidate
    return found


class LabelArchive:
    """TL.zip / VL.zip(또는 압축 해제 디렉토리)에서 라벨 CSV를 읽는다."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise MatchError(f"라벨링 데이터를 찾을 수 없습니다: {self.path}", self.path.parent)
        self._zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self._cached_names: frozenset[str] | None = None

    def _names(self) -> frozenset[str]:
        if self._cached_names is not None:
            return self._cached_names
        self._cached_names = self._build_names()
        return self._cached_names

    def _build_names(self) -> frozenset[str]:
        if self._zip is None:
            root = self.path
            return frozenset(
                nfc(str(p.relative_to(root)).replace("\\", "/")) for p in root.rglob("*.csv")
            )
        return frozenset(nfc(name) for name in self._zip.namelist())

    def has(self, name: str) -> bool:
        return nfc(name) in self._names()

    def read_csv(self, name: str):
        import pandas as pd

        if not self.has(name):
            raise MatchError(
                f"대조할 라벨 CSV가 없습니다:\n{name}\n"
                f"라벨링 데이터: {self.path}",
                self.path.parent,
            )
        if self._zip is None:
            return pd.read_csv(self.path / name)
        # zip 내부 실제 키는 NFD일 수 있으므로 정규화해서 되찾는다.
        actual = next((n for n in self._zip.namelist() if nfc(n) == nfc(name)), name)
        return pd.read_csv(io.BytesIO(self._zip.read(actual)))

    def ground_truth_common3d(self, clip: SourceClip) -> np.ndarray:
        """(T, 18, 3) 미터 단위 Common Skeleton 정답 좌표."""
        frame = self.read_csv(clip.ground_truth_3d)
        columns = [f"{name}_{axis}" for name in JOINT_NAMES for axis in ("x", "y", "z")]
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise MatchError(
                f"정답 CSV에 없는 관절 열이 있습니다: {missing[:4]} …\n{clip.ground_truth_3d}",
                self.path.parent,
            )
        values = frame[columns].to_numpy(dtype=float)
        coords = values.reshape(len(frame), len(JOINT_NAMES), 3) / 1000.0  # mm -> m
        return coords[:, COMMON_FROM_AIHUB_IDX, :]


def _hip_center(coords: np.ndarray) -> np.ndarray:
    return coords - coords[..., _HIP : _HIP + 1, :]


def _procrustes_align(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """회전·균일 스케일만 맞춘다(반사는 금지) — 좌표계가 달라 생기는 차이를 제거.

    사람을 못 찾은 프레임은 NaN으로 들어오는데, 그대로 SVD에 넣으면 수렴하지 않는다.
    정합할 수 없는 프레임은 NaN을 그대로 돌려보내 집계에서 빠지게 한다.
    """
    if not (np.isfinite(predicted).all() and np.isfinite(target).all()):
        return np.full_like(predicted, np.nan)
    pred = predicted - predicted.mean(axis=0, keepdims=True)
    ref = target - target.mean(axis=0, keepdims=True)
    try:
        u, s, vt = np.linalg.svd(pred.T @ ref)
    except np.linalg.LinAlgError:
        return np.full_like(predicted, np.nan)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # 반사가 섞이면 좌우가 뒤집힌다
        u[:, -1] *= -1
        rotation = u @ vt
    scale = s.sum() / max(float((pred ** 2).sum()), 1e-12)
    return pred @ rotation * scale + target.mean(axis=0, keepdims=True)


@dataclass(frozen=True)
class MatchReport:
    """한 시점의 대조 결과."""

    view: str
    frames: int
    detected_frames: int  # 사람을 찾아 실제로 대조한 프레임 수
    mpjpe_mm: float          # 골반 정렬만 한 평균 관절 오차
    pa_mpjpe_mm: float       # 회전·스케일까지 맞춘 뒤의 오차
    match_rate: float        # PCK: 허용 오차 안에 든 관절 비율(%)
    threshold_mm: float
    per_joint_mm: dict[str, float]

    def summary(self) -> str:
        return (f"{VIEW_LABEL_KO.get(self.view, self.view)} "
                f"매칭율 {self.match_rate:.1f}% · 오차 {self.pa_mpjpe_mm:.0f}mm "
                f"({self.detected_frames}/{self.frames}프레임)")


def compare_sequences(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    view: str,
    threshold_mm: float = 100.0,
) -> MatchReport:
    """(T,18,3) 추정과 정답을 프레임 수에 맞춰 잘라 좌표 매칭율을 낸다.

    두 좌표계가 서로 다르므로(우리 것은 카메라 기준, 정답은 캡처 스튜디오 기준)
    골반을 원점으로 맞춘 뒤 회전·스케일을 Procrustes로 정합한 값을 주 지표로 쓴다.
    match_rate는 3D 포즈 평가에서 흔히 쓰는 PCK — 정답에서 threshold 안에 든 관절 비율이다.
    """
    length = min(len(predicted), len(ground_truth))
    if length == 0:
        raise MatchError("대조할 프레임이 없습니다 (추정 또는 정답 시퀀스가 비어 있습니다)")
    pred = _hip_center(np.asarray(predicted, dtype=float)[:length])
    truth = _hip_center(np.asarray(ground_truth, dtype=float)[:length])

    raw_error = np.linalg.norm(pred - truth, axis=-1) * 1000.0
    aligned = np.stack([_procrustes_align(pred[i], truth[i]) for i in range(length)])
    aligned_error = np.linalg.norm(aligned - truth, axis=-1) * 1000.0

    valid = np.isfinite(aligned_error)
    detected = int(np.isfinite(aligned_error).all(axis=1).sum())
    if not valid.any():
        raise MatchError("모든 프레임에서 사람을 찾지 못해 대조할 수 없습니다")
    return MatchReport(
        view=view,
        frames=length,
        detected_frames=detected,
        mpjpe_mm=float(np.nanmean(raw_error)),
        pa_mpjpe_mm=float(np.nanmean(aligned_error)),
        match_rate=100.0 * float(np.mean(aligned_error[valid] <= threshold_mm)),
        threshold_mm=threshold_mm,
        per_joint_mm={
            name: float(np.nanmean(aligned_error[:, i]))
            for i, name in enumerate(COMMON_JOINT_NAMES)
        },
    )


__all__ = [
    "ERROR_TYPES", "LEVELS", "LABEL_PREFIX", "VIEW_CAMERAS", "VIEW_LABEL_KO",
    "LabelArchive", "MatchError", "MatchReport", "SourceClip",
    "compare_sequences", "find_view_video", "nfc", "parse_source_video", "sibling_view_videos",
]


@dataclass(frozen=True)
class TakeEntry:
    """드롭다운에 채울 한 개의 촬영분(배우 + 트라이)."""

    error_type: str
    level: str
    actor: str
    rep: str
    videos: dict[str, Path]  # view -> 영상 경로 (없는 시점은 빠진다)

    @property
    def label(self) -> str:
        views = "".join(VIEW_LABEL_KO[v][0] for v in ("front", "oblique", "side") if v in self.videos)
        return f"{self.actor} · 트라이 {self.rep}  [{views or '영상없음'}]"


def scan_source_root(root: str | Path) -> dict[tuple[str, str], list[TakeEntry]]:
    """원천데이터 루트를 훑어 (클래스, 난이도) -> 촬영분 목록을 만든다.

    트라이는 1~4가 다 있는 경우가 오히려 드물다(실측: 176개 배우-조합 중 17개만 완비).
    그래서 고정 목록을 쓰지 않고 실제로 있는 폴더만 담는다.
    """
    base = Path(root)
    if not base.is_dir():
        raise MatchError(f"원천데이터 폴더를 찾을 수 없습니다: {base}", base.parent)

    catalog: dict[tuple[str, str], list[TakeEntry]] = {}
    for error_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        error_type = nfc(error_dir.name)
        for level_dir in sorted(p for p in error_dir.iterdir() if p.is_dir()):
            level = nfc(level_dir.name)
            entries: list[TakeEntry] = []
            for actor_dir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
                actor = nfc(actor_dir.name)
                for rep_path in sorted(p for p in actor_dir.iterdir() if p.is_dir()):
                    rep = nfc(rep_path.name)
                    rep_dir = rep_path
                    videos = {}
                    for view, camera in VIEW_CAMERAS.items():
                        candidate = find_view_video(rep_dir, camera, rep)
                        if candidate is not None:
                            videos[view] = candidate
                    if videos:
                        entries.append(TakeEntry(error_type, level, actor, rep, videos))
            if entries:
                catalog[(error_type, level)] = entries
    if not catalog:
        raise MatchError(
            f"원천데이터를 찾지 못했습니다: {base}\n"
            f"기대 구조: {base.name}/{{클래스}}/{{난이도}}/{{배우}}/{{트라이}}/camera{{N}}/video/*.avi",
            base,
        )
    return catalog


LABEL_ARCHIVE_NAMES = ("TL.zip", "VL.zip")
_ARCHIVE_CACHE: dict[str, LabelArchive] = {}


def get_archive(path: str | Path) -> LabelArchive:
    """같은 ZIP을 여러 번 열지 않도록 캐시해서 돌려준다(항목이 13만 개라 매번 읽으면 느리다)."""
    key = str(Path(path).resolve())
    if key not in _ARCHIVE_CACHE:
        _ARCHIVE_CACHE[key] = LabelArchive(path)
    return _ARCHIVE_CACHE[key]


def find_label_archives(extra_roots: Iterable[str | Path] = ()) -> list[Path]:
    """TL.zip / VL.zip을 흔한 위치에서 찾는다. 사용자가 고르지 않아도 되게 하기 위함.

    외장 하드나 홈 전체를 재귀 탐색하면 수십 초가 걸리므로, 깊이를 제한한 glob만 쓴다.
    AI Hub가 내려주는 폴더 구조가 고정이라 이 정도로 충분하다.
    """
    roots = [Path(r) for r in extra_roots]
    roots += [
        Path.home() / "Project" / "Crossfit_Labeling_Data",
        Path.home() / "Project",
        Path.home() / "Desktop",
        Path.home() / "Downloads",
    ]
    source_root = find_source_root()
    if source_root is not None:
        roots.append(source_root.parent)

    # AI Hub 기본 구조: {루트}/213.크로스핏_동작_데이터/01-1.정식개방데이터/{Training|Validation}/02.라벨링데이터/TL.zip
    patterns = ("*.zip", "*/*.zip", "*/*/*.zip", "*/*/*/*.zip", "*/*/*/*/*.zip")
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if path.name in LABEL_ARCHIVE_NAMES and path not in found:
                    found.append(path)
    return found


def resolve_ground_truth(clip: SourceClip,
                         candidates: Iterable[str | Path]) -> LabelArchive | None:
    """이 촬영분의 3D 정답이 실제로 들어 있는 아카이브를 고른다.

    한 배우/트라이는 Training(TL) 또는 Validation(VL) 중 한쪽에만 있으므로, 둘 다 열어
    실제로 가진 쪽을 쓴다.
    """
    for candidate in candidates:
        try:
            archive = get_archive(candidate)
        except MatchError:
            continue
        if archive.has(clip.ground_truth_3d):
            return archive
    return None


DEFAULT_SOURCE_ROOTS = (
    Path("/Volumes/My Passport/에어스쿼트"),   # macOS
    Path("Z:/에어스쿼트"),                      # Windows (외장 하드가 Z드라이브)
)


def find_source_root() -> Path | None:
    """연결된 외장 하드에서 원천데이터 루트를 찾는다 (윈도우/맥 공통)."""
    for candidate in DEFAULT_SOURCE_ROOTS:
        if candidate.is_dir():
            return candidate
    for volume in Path("/Volumes").glob("*/에어스쿼트"):
        if volume.is_dir():
            return volume
    return None


__all__ = __all__ + [
    "TakeEntry", "scan_source_root", "find_source_root", "DEFAULT_SOURCE_ROOTS",
    "LABEL_ARCHIVE_NAMES", "find_label_archives", "get_archive", "resolve_ground_truth",
]
