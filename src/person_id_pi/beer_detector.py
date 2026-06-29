from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple

import numpy as np

from .beverage_detector import BeverageDetector
from .beverage_types import BeverageDetection, BeverageLabel


@dataclass(frozen=True)
class BeerDetection:
    bbox: Tuple[int, int, int, int]
    category: BeverageLabel
    score: float


class BeerDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[BeerDetection]: ...


class FilteredBeerDetector:
    """Turn generic COCO cup/bottle proposals into weekend-v1 beer candidates."""

    def __init__(
        self,
        detector: BeverageDetector,
        can_aspect_ratio_min: float = 0.65,
        can_aspect_ratio_max: float = 2.4,
    ) -> None:
        self.detector = detector
        self.can_aspect_ratio_min = can_aspect_ratio_min
        self.can_aspect_ratio_max = can_aspect_ratio_max

    def detect(self, frame: np.ndarray) -> List[BeerDetection]:
        accepted: List[BeerDetection] = []
        for detection in self.detector.detect(frame):
            x1, y1, x2, y2 = detection.bbox
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            if detection.label == "cup":
                accepted.append(
                    BeerDetection(
                        bbox=detection.bbox,
                        category="cup",
                        score=detection.score,
                    )
                )
                continue
            if detection.label not in {"bottle", "can"}:
                continue
            aspect_ratio = height / width
            if (
                not self.can_aspect_ratio_min
                <= aspect_ratio
                <= self.can_aspect_ratio_max
            ):
                continue
            accepted.append(
                BeerDetection(
                    bbox=detection.bbox,
                    category="can",
                    score=detection.score,
                )
            )
        return accepted
