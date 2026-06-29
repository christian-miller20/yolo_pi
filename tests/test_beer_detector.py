import numpy as np

from person_id_pi.beer_detector import FilteredBeerDetector
from person_id_pi.beverage_types import BeverageDetection


class _Detector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


def test_accepts_all_cups_and_maps_can_shaped_bottles():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[10:50, 10:40] = (0, 0, 255)
    detections = [
        BeverageDetection((10, 10, 40, 50), "cup", 0.9),
        BeverageDetection((50, 10, 80, 50), "cup", 0.8),
        BeverageDetection((90, 10, 110, 42), "bottle", 0.7),
    ]
    detector = FilteredBeerDetector(_Detector(detections))

    result = detector.detect(frame)

    assert [item.category for item in result] == ["cup", "cup", "can"]


def test_rejects_tall_bottle_shape():
    frame = np.zeros((120, 100, 3), dtype=np.uint8)
    detector = FilteredBeerDetector(
        _Detector([BeverageDetection((10, 5, 25, 100), "bottle", 0.9)])
    )

    assert detector.detect(frame) == []
