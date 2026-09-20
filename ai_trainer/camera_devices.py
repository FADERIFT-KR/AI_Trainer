"""카메라 선택 화면에서 사용할 로컬 영상 장치 열거."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDevice:
    index: int
    description: str
    device_name: str = ""
    verified: bool = True

    @property
    def button_label(self) -> str:
        suffix = "" if self.verified else " · 기본값"
        return f"카메라 {self.index} · {self.description}{suffix}"


def default_camera_index() -> int:
    try:
        value = int(os.environ.get("AI_TRAINER_CAMERA_INDEX", "0"))
    except ValueError:
        return 0
    return max(0, value)


def discover_camera_devices() -> list[CameraDevice]:
    """Return Qt multimedia devices in the same order used as OpenCV indexes.

    Qt exposes friendly DirectShow/AVFoundation descriptions without opening
    every camera.  Some minimal/headless installations expose no multimedia
    devices; in that case the configured OpenCV index remains selectable and
    is marked as unverified rather than blocking the application.
    """
    found: list[CameraDevice] = []
    try:
        from PyQt5.QtMultimedia import QCameraInfo

        for index, camera in enumerate(QCameraInfo.availableCameras()):
            description = camera.description() or f"영상 장치 {index}"
            device_name = camera.deviceName()
            if not isinstance(device_name, str):
                try:
                    device_name = bytes(device_name).decode(errors="replace")
                except (TypeError, ValueError):
                    device_name = str(device_name)
            found.append(CameraDevice(index, description, device_name, True))
    except (ImportError, RuntimeError):
        pass

    if found:
        return found
    index = default_camera_index()
    return [CameraDevice(index, "OpenCV 영상 장치", verified=False)]


__all__ = ["CameraDevice", "default_camera_index", "discover_camera_devices"]
