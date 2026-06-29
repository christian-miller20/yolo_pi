import numpy as np

from person_id_pi.beverage_config import BeverageDetectorConfig
from person_id_pi.beverage_detector import YoloBeverageDetector


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)


class _FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _FakeModel:
    def __init__(self, results):
        self._results = results

    def __call__(self, frame, verbose=False):
        return self._results


def test_yolo_detector_maps_and_filters_classes():
    names = {
        0: "cup",
        1: "bottle",
        2: "person",
        3: "beer can",
    }
    boxes = _FakeBoxes(
        xyxy=[[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30], [30, 30, 40, 40]],
        conf=[0.9, 0.1, 0.95, 0.8],
        cls=[0, 1, 2, 3],
    )
    model = _FakeModel([_FakeResult(names=names, boxes=boxes)])
    detector = YoloBeverageDetector(
        config=BeverageDetectorConfig(conf_min=0.25),
        model=model,
    )

    detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(detections) == 2
    assert detections[0].label == "cup"
    assert detections[1].label == "can"
    assert detections[0].bbox == (0, 0, 10, 10)
    assert detections[1].bbox == (30, 30, 40, 40)


def test_yolo_detector_respects_allowed_labels():
    names = {
        0: "cup",
        1: "bottle",
    }
    boxes = _FakeBoxes(
        xyxy=[[0, 0, 10, 10], [10, 10, 20, 20]],
        conf=[0.9, 0.9],
        cls=[0, 1],
    )
    model = _FakeModel([_FakeResult(names=names, boxes=boxes)])
    detector = YoloBeverageDetector(
        config=BeverageDetectorConfig(conf_min=0.25, allowed_labels=("cup",)),
        model=model,
    )

    detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "cup"
