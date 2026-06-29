from __future__ import annotations

from typing import Protocol, Tuple

import cv2
import numpy as np

from .live_config import CameraConfig


class FrameSource(Protocol):
    def read(self) -> Tuple[bool, np.ndarray | None]: ...

    def close(self) -> None: ...


class OpenCVCameraSource:
    """OpenCV camera adapter for local development and camera smoke tests."""

    def __init__(self, index: int, config: CameraConfig) -> None:
        self._capture = cv2.VideoCapture(index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._capture.set(cv2.CAP_PROP_FPS, config.fps)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Unable to open camera index {index}")

    def read(self) -> Tuple[bool, np.ndarray | None]:
        ok, frame = self._capture.read()
        return bool(ok), frame if ok else None

    def close(self) -> None:
        self._capture.release()


class PiCameraSource:
    """Small Picamera2 adapter; imports camera libraries only on the Pi."""

    def __init__(self, config: CameraConfig) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Picamera2 is required for live capture on Raspberry Pi OS"
            ) from exc
        self._camera = Picamera2()
        camera_config = self._camera.create_video_configuration(
            main={
                "size": (config.width, config.height),
                "format": "RGB888",
            },
            controls={"FrameRate": config.fps},
        )
        self._camera.configure(camera_config)
        self._camera.start()

    def read(self) -> Tuple[bool, np.ndarray | None]:
        frame_rgb = self._camera.capture_array("main")
        if frame_rgb is None:
            return False, None
        return True, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self._camera.stop()
        self._camera.close()
