"""Optional local trace; coordinates and lossless pre-overlay observation images."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np


class PoseDiagnostics:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")
        self.path = path
        self.image_dir = path.parent / (path.stem + "_images_" + uuid4().hex)
        self.last_observation_id = -1

    def write_observation(self, image_bgr, *, observation_id, mirrored, **record):
        """Commit image first, then its JSON row. On failure stop, never silently skip."""
        try:
            import cv2

            if not isinstance(observation_id, int) or observation_id <= self.last_observation_id:
                raise ValueError("observation_id must be strictly increasing integers")
            if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
                raise ValueError("Expected uint8 BGR observation image")
            if record.get("image_size") != [image_bgr.shape[1], image_bgr.shape[0]]:
                raise ValueError("image_size does not match observation image")
            ok, encoded = cv2.imencode(".png", image_bgr)
            if not ok:
                raise OSError("PNG encoding failed")
            if not self.image_dir.exists():
                self.image_dir.mkdir(exist_ok=False)
            target = self.image_dir / f"observation_{observation_id:08d}.png"
            payload = encoded.tobytes()
            with target.open("xb") as stream:
                stream.write(payload)
            self.write(
                **record, observation_id=observation_id,
                observation_image={
                    "observation_id": observation_id,
                    "path": target.relative_to(self.path.parent).as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "mirrored": bool(mirrored), "orientation": "mediapipe_input",
                    "overlay": False, "format": "png",
                },
            )
            self.last_observation_id = observation_id
        except Exception as exc:
            raise RuntimeError(
                f"진단 이미지/좌표 저장 실패 ({self.path}, observation_id={observation_id}). "
                f"진단 및 현재 게임 세션을 중단합니다. 마지막 불완전 파일이 남을 수 있습니다: {exc}"
            ) from exc

    def write(self, **record):
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)

        result = record.get("completed_rep")
        if result is not None:
            record["completed_rep"] = asdict(result)
        self.stream.write(json.dumps(record, default=encode, ensure_ascii=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()
