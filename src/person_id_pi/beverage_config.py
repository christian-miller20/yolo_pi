from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BeverageDetectorConfig:
    # YOLO model identifier or local path.
    model_path: str = "yolov8n.pt"
    # Minimum confidence for beverage detections.
    conf_min: float = 0.25
    allowed_labels: Tuple[str, ...] = ("cup", "can", "bottle")
