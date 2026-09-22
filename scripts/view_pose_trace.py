"""Read-only, offline front/side viewer for game PoseDiagnostics JSONL files."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_pose_trace import raw_common
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.game_ui.skeleton_panel import SkeletonViews

IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


@dataclass
class Observation:
    record: dict
    world: np.ndarray | None
    raw: np.ndarray | None
    common: np.ndarray | None
    aligned: np.ndarray | None
    errors: dict[str, str]


def coordinates(value, shape):
    if value is None:
        return None, "기록 없음 / null"
    try:
        arr = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return None, "숫자 배열이 아닙니다"
    if arr.shape != shape:
        return None, f"형태 오류: {arr.shape}, 필요: {shape}"
    if not np.isfinite(arr).all():
        return None, "NaN / Infinity / null 등 유효하지 않은 값"
    return arr, ""


def observation(record):
    errors = {}
    world, errors["world"] = coordinates(record.get("world_landmarks"), (33, 4))
    raw = raw_common(world[None])[0] if world is not None else None
    errors["raw"] = errors["world"]
    common, errors["common"] = coordinates(record.get("common3d"), (18, 3))
    aligned, errors["aligned"] = coordinates(record.get("aligned_frame"), (18, 3))
    return Observation(record, world, raw, common, aligned, errors)


def load_trace(path: Path):
    """Keep physical line numbers intact; reject corrupt lines rather than skip them."""
    rows = []
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"{path}: 파일 {line_number}행이 비어 있습니다 (0-based {line_number-1})")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}: 파일 {line_number}행 JSON 손상: {exc.msg}, 열 {exc.colno}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}: 파일 {line_number}행은 JSON 객체가 아닙니다")
                rows.append(observation(record))
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"로그 읽기 실패: {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"빈 JSONL 파일입니다: {path}")
    return rows


def angle(a, b, c):
    u, v = a-b, c-b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v)/denom, -1, 1))))


def metrics(coords):
    if coords is None:
        return np.full(6, np.nan)
    result = []
    for side in ("L", "R"):
        hip, knee, ankle = (coords[IDX[side+n]] for n in ("Hip", "Knee", "Ankle"))
        result.extend((angle(hip, knee, ankle), angle(coords[IDX["Neck"]], hip, knee),
                       float(np.linalg.norm(hip-knee))))
    return np.array(result)


def number(value, precision=3):
    return f"{value:.{precision}f}" if np.isfinite(value) else "표시 불가"


def frozen_text(value):
    if not isinstance(value, list) or len(value) != 18 or not all(isinstance(v, bool) for v in value):
        return "표시 불가 (Boolean 18개 기록 필요)"
    names = [n for n, frozen in zip(COMMON_JOINT_NAMES, value) if frozen]
    return f"{sum(value)}/18: " + (", ".join(names) if names else "없음")


def load_observation_image(record, trace_path):
    """Read only the explicitly linked local image; never guess a neighboring frame."""
    import cv2

    meta = record.get("observation_image")
    if meta is None:
        return None, "이미지 기록 없음 (기존 좌표 전용 로그)"
    try:
        if not isinstance(meta, dict) or type(record.get("observation_id")) is not int:
            raise ValueError("관측 식별자 형식 오류")
        if meta.get("observation_id") != record["observation_id"]:
            raise ValueError("이미지와 좌표의 observation_id 불일치")
        relative = Path(meta["path"])
        root = trace_path.resolve().parent
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):
            raise ValueError("로그 폴더 밖의 이미지 경로는 열지 않습니다")
        payload = target.read_bytes()
        if hashlib.sha256(payload).hexdigest() != meta.get("sha256"):
            raise ValueError("이미지 SHA-256 불일치 (파일 변경/손상)")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("PNG 읽기 실패")
        if [image.shape[1], image.shape[0]] != record.get("image_size"):
            raise ValueError("이미지 크기와 좌표 기록 불일치")
        if meta.get("orientation") != "mediapipe_input" or meta.get("overlay") is not False:
            raise ValueError("MediaPipe 입력 방향/오버레이 여부 확인 불가")
        return image, f"observation_id={record['observation_id']} | MediaPipe 입력과 동일 방향 | 좌우 반전={meta.get('mirrored')}"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return None, f"이미지 표시 불가: {exc}"


def create_window(rows, path, initial=0):
    # Qt imports stay out of loading/metric helpers, allowing non-GUI inspection.
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPixmap, QKeySequence
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                                QHeaderView, QShortcut, QScrollArea, QCheckBox)

    class Viewer(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"3D 좌표 진단 (읽기 전용) — {path.name}")
            outer = QVBoxLayout(self)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            body = QWidget()
            layout = QVBoxLayout(body)
            scroll.setWidget(body)
            outer.addWidget(scroll)
            title = QLabel(str(path.resolve()) + "\n정규화 전후는 시각 차이가 있을 수 있는 참고 비교입니다. 보간 시각은 로그에 없습니다.")
            title.setWordWrap(True)
            layout.addWidget(title)
            navigation = QHBoxLayout()
            self.index = QSpinBox()
            self.index.setRange(0, len(rows)-1)
            self.index.setPrefix("관측 행 (0-based): ")
            self.index.setKeyboardTracking(False)
            previous, following = QPushButton("← 이전"), QPushButton("다음 →")
            previous.clicked.connect(lambda: self.index.setValue(self.index.value()-1))
            following.clicked.connect(lambda: self.index.setValue(self.index.value()+1))
            navigation.addWidget(previous)
            navigation.addWidget(self.index)
            navigation.addWidget(following)
            navigation.addWidget(QLabel(f"범위 0–{len(rows)-1} | 파일 줄 번호는 +1 | sample_index와 다름"))
            layout.addLayout(navigation)
            for key, delta in (("Left", -1), ("Right", 1)):
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(lambda d=delta: self.index.setValue(self.index.value()+d))
            self.info = QLabel()
            self.info.setWordWrap(True)
            self.info.setTextFormat(Qt.PlainText)
            self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(self.info)
            self.image_info = QLabel()
            self.image_info.setWordWrap(True)
            self.image_info.setTextFormat(Qt.PlainText)
            layout.addWidget(self.image_info)
            camera_panels = QHBoxLayout()
            self.camera_images = []
            for title in ("동일 관측: 원본 이미지 (오버레이 없음)",
                          "동일 관측: Image Landmarks (visibility ≥ 0.45)",
                          "동일 관측: World → 18관절 XY / ZY"):
                box = QVBoxLayout()
                box.addWidget(QLabel(title))
                label = QLabel()
                label.setFixedSize(480, 300)
                label.setAlignment(Qt.AlignCenter)
                label.setWordWrap(True)
                box.addWidget(label)
                self.camera_images.append(label)
                camera_panels.addLayout(box)
            layout.addLayout(camera_panels)
            # Fixed transforms over the entire trace, never per-frame auto zoom.
            raw_common_seq = [a for r in rows for a in (r.raw, r.common) if a is not None]
            aligned_seq = [r.aligned for r in rows if r.aligned is not None]
            self.raw_view = SkeletonViews(np.stack(raw_common_seq), 480, 300) if raw_common_seq else None
            self.aligned_view = SkeletonViews(np.stack(aligned_seq), 480, 300) if aligned_seq else None
            # Viewer-local transform wrapper: reflect screen geometry, not the
            # source coordinates or the canvas (which would invert text too).
            self.flipped_raw_view = None
            if self.raw_view is not None:
                from copy import copy
                self.flipped_raw_view = copy(self.raw_view)
                def flipped_transform(points):
                    pixels = self.raw_view.transform(points).copy()
                    pixels[..., 1] = self.raw_view.height - pixels[..., 1]
                    return pixels
                self.flipped_raw_view.transform = flipped_transform
            self.flip_world_y = QCheckBox("World Y 표시 뒤집기 (−Y 위쪽)")
            layout.addWidget(self.flip_world_y)
            self.y_direction = QLabel()
            self.y_direction.setWordWrap(True)
            layout.addWidget(self.y_direction)
            panels = QHBoxLayout()
            self.images = []
            for text in (
                "① World 원본 33 → ② 매핑만 한 18관절\nMediaPipe XYZ · m · 회전/필터 없음",
                "③ common3d (로그 그대로)\n필터·저신뢰 유지·Hip 중심 이동 후 · m",
                "④ aligned_frame (로그 그대로)\n몸 기준 XYZ · 다리 길이로 나눈 무차원\n별도 화면 스케일 · 마지막 보간 샘플",
            ):
                box = QVBoxLayout()
                label = QLabel(text)
                label.setWordWrap(True)
                box.addWidget(label)
                image = QLabel()
                image.setFixedSize(480, 300)
                image.setAlignment(Qt.AlignCenter)
                image.setWordWrap(True)
                box.addWidget(image)
                self.images.append(image)
                panels.addLayout(box)
            layout.addLayout(panels)
            legend = QLabel("좌측 2패널은 전체 로그에 걸친 동일 고정 스케일·변환입니다. Hip 중심 이동 차이는 그대로 표시합니다.\n"
                            "가로 +X(정면) / +Z(측면). aligned_frame은 항상 +Y 위쪽입니다.\n"
                            "L=모델의 왼쪽 관절(파랑 계열), R=오른쪽(빨강 계열). 무릎·엉덩이에 L/R 라벨 표시.")
            legend.setWordWrap(True)
            layout.addWidget(legend)
            self.stats = QTableWidget(6, 4)
            self.stats.setHorizontalHeaderLabels(["항목", "World 매핑", "필터 후", "필터 − 원본"])
            self.stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.stats.setMinimumHeight(220)
            layout.addWidget(self.stats)
            self.length_info = QLabel()
            layout.addWidget(self.length_info)
            self.world_table = QTableWidget(33, 5)
            self.world_table.setHorizontalHeaderLabels(["MP index", "X (m)", "Y (m)", "Z (m)", "visibility"])
            self.world_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            layout.addWidget(QLabel("① 원본 World 33관절 수치 (매핑 전, 기록 그대로)"))
            self.world_table.setMinimumHeight(240)
            layout.addWidget(self.world_table)
            for table in (self.stats, self.world_table):
                table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.index.valueChanged.connect(self.refresh)
            self.flip_world_y.toggled.connect(lambda _: self.refresh(self.index.value()))
            self.index.setValue(initial)
            self.refresh(initial)
            self.resize(1530, 950)

        def refresh(self, index):
            import cv2
            row = rows[index]
            record = row.record
            flipped = self.flip_world_y.isChecked()
            world_view = self.flipped_raw_view if flipped else self.raw_view
            self.y_direction.setText(
                "World / common3d: " + ("−Y 위쪽, +Y 아래쪽" if flipped else "+Y 위쪽, −Y 아래쪽 (기존 표시)")
                + " | 표시 방향만 변경합니다. 웹캠과 동일한 원근·시점을 복원하지 않습니다."
            )
            completed = record.get('completed_rep')
            summary = ({k: completed.get(k) for k in ('rep_index', 'predicted_class', 'classifier_source', 'frame_range')}
                       if isinstance(completed, dict) else completed)
            self.info.setText(
                f"파일 줄 {index+1} / 관측 행 {index} | timestamp={record.get('timestamp', '기록 없음')} s (관측 시각)"
                f" | observation_id={record.get('observation_id', '기록 없음')}"
                f" | sample_index={record.get('sample_index', '기록 없음')} | phase={record.get('phase', '기록 없음')}\n"
                f"frozen3d: {frozen_text(record.get('frozen3d'))}\n"
                f"완료 REP 요약 (재판정 없음): {json.dumps(summary, ensure_ascii=False)}"
            )
            camera, message = load_observation_image(record, path)
            self.image_info.setText(message)
            for widget in self.camera_images:
                widget.clear()
            if camera is None:
                for widget in self.camera_images[:2]:
                    widget.setText(message)
            else:
                def show_camera(widget, frame):
                    rgb = np.ascontiguousarray(frame[:, :, ::-1])
                    img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
                    widget.setPixmap(QPixmap.fromImage(img).scaled(480, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                show_camera(self.camera_images[0], camera)
                landmarks, error = coordinates(record.get('image_landmarks'), (33, 4))
                if landmarks is None:
                    self.camera_images[1].setText('2D 표시 불가: ' + error)
                else:
                    from ai_trainer.live_pose.render import draw_2d_pose
                    show_camera(self.camera_images[1], draw_2d_pose(camera, landmarks))
            for widget, coords, key, view in zip(self.images, (row.raw, row.common, row.aligned),
                                                ('raw', 'common', 'aligned'),
                                                (world_view, world_view, self.aligned_view)):
                widget.clear()
                if coords is None:
                    widget.setText("표시 불가\n" + row.errors[key])
                    continue
                canvas = view.render(coords, estimated=True)
                for panel, axes in enumerate(([0, 1], [2, 1])):
                    projected = view.transform(coords[:, axes])
                    for name in ("LHip", "RHip", "LKnee", "RKnee"):
                        x, y = projected[IDX[name]]
                        cv2.putText(canvas, name, (int(x)+panel*240+4, int(y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
                rgb = np.ascontiguousarray(canvas[:, :, ::-1])
                qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
                widget.setPixmap(QPixmap.fromImage(qimage))
            if self.images[0].pixmap() is not None:
                self.camera_images[2].setPixmap(self.images[0].pixmap())
            else:
                self.camera_images[2].setText(self.images[0].text())
            raw, filtered = metrics(row.raw), metrics(row.common)
            labels = ["L 무릎각 (°)", "L 고관절각 Neck–Hip–Knee (°)", "L Hip–Knee (m)",
                      "R 무릎각 (°)", "R 고관절각 Neck–Hip–Knee (°)", "R Hip–Knee (m)"]
            for i, label in enumerate(labels):
                for col, text in enumerate((label, number(raw[i]), number(filtered[i]), number(filtered[i]-raw[i]))):
                    self.stats.setItem(i, col, QTableWidgetItem(text))
            relative = (filtered[5]/raw[5]-1)*100 if np.isfinite(raw[5]) and raw[5] > 1e-12 else float('nan')
            initial_raw = metrics(rows[0].raw)[5]
            initial_filtered = metrics(rows[0].common)[5]
            dr, df = raw[5]-initial_raw, filtered[5]-initial_filtered
            self.length_info.setText(f"오른쪽 Hip–Knee: 필터 − 원본 {number(filtered[5]-raw[5])} m ({number(relative)}%)"
                                     f" | 관측 0 대비 변화: 원본 {number(dr)} m / 필터 {number(df)} m")
            for i in range(33):
                values = [str(i)] + ([number(v, 5) for v in row.world[i]] if row.world is not None else ["표시 불가"]*4)
                for col, text in enumerate(values):
                    self.world_table.setItem(i, col, QTableWidgetItem(text))

    return Viewer()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="읽기 전용 JSONL 경로")
    parser.add_argument("--row", type=int, default=0, help="시작 관측 행 (0-based; sample_index 아님)")
    args = parser.parse_args(argv)
    try:
        rows = load_trace(args.trace)
        if not 0 <= args.row < len(rows):
            raise ValueError(f"--row 범위는 0–{len(rows)-1}입니다")
    except ValueError as exc:
        print(f"진단 로그 오류: {exc}", file=sys.stderr)
        return 2
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv[:1])
    window = create_window(rows, args.trace, args.row)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
