from datetime import datetime, timezone

import numpy as np

from person_id_pi.beer_detector import BeerDetection
from person_id_pi.beverage_store import BeverageStore
from person_id_pi.face_types import FaceDetection, FaceEmbedding
from person_id_pi.identity_config import IdentityConfig
from person_id_pi.identity_engine import IdentityEngine
from person_id_pi.identity_store import IdentityStore
from person_id_pi.live_config import (
    CameraConfig,
    EventConfig,
    LiveConfig,
    MotionConfig,
    StorageConfig,
)
from person_id_pi.live_service import LiveService, MotionDetector


class _Source:
    def __init__(self):
        self.closed = False

    def read(self):
        return False, None

    def close(self):
        self.closed = True


class _FaceEmbedder:
    def detect(self, frame):
        return [
            FaceDetection(
                bbox=(0, 0, 20, 20),
                landmarks=None,
                quality=1.0,
                det_score=1.0,
                size_score=1.0,
                blur_score=1.0,
                aligned_crop=frame[0:20, 0:20],
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )
        ]

    def embed(self, face):
        return FaceEmbedding(
            embedding=face.embedding,
            quality=face.quality,
            bbox=face.bbox,
            landmarks=None,
        )


class _BeerDetector:
    def detect(self, frame):
        if not np.any(frame):
            return []
        return [BeerDetection((20, 15, 40, 45), "cup", 0.9)]


def test_motion_detector_requires_sustained_samples():
    detector = MotionDetector(
        20, 20, pixel_delta=10, changed_ratio=0.2, sustained_samples=2
    )
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    white = np.full((20, 20, 3), 255, dtype=np.uint8)

    assert detector.update(black)[1] is False
    assert detector.update(white)[1] is False
    moving, triggered, ratio = detector.update(black)
    assert moving is True
    assert triggered is True
    assert ratio == 1.0


def test_live_event_auto_enrolls_and_counts(tmp_path):
    source = _Source()
    config = LiveConfig(
        camera=CameraConfig(width=80, height=60, fps=5),
        motion=MotionConfig(
            width=20,
            height=20,
            sample_fps=5,
            pixel_delta=10,
            changed_ratio=0.2,
            sustained_samples=1,
        ),
        event=EventConfig(
            pre_roll_seconds=1.0,
            quiet_seconds=0.5,
            max_seconds=5,
            inference_stride=1,
            object_min_seen_frames=2,
            cooldown_seconds=600,
        ),
        storage=StorageConfig(
            identity_store=str(tmp_path / "identities.json"),
            beer_store=str(tmp_path / "beers.json"),
            evidence_dir=str(tmp_path / "events"),
            decision_log=str(tmp_path / "decisions.jsonl"),
            record_evidence=False,
        ),
    )
    identity_store = IdentityStore(config.storage.identity_store)
    identity = IdentityEngine(
        identity_store,
        IdentityConfig(n_min=1, quality_min=0.1, accept_threshold=0.5),
    )
    beer_store = BeverageStore(config.storage.beer_store)
    service = LiveService(
        config,
        source,
        _FaceEmbedder(),
        identity,
        _BeerDetector(),
        beer_store,
        log=lambda message: None,
        utcnow=lambda: datetime(2026, 6, 26, tzinfo=timezone.utc),
        async_inference=False,
    )
    black = np.zeros((60, 80, 3), dtype=np.uint8)
    white = np.full((60, 80, 3), 255, dtype=np.uint8)

    assert service.process_frame(black, 0.0) is None
    assert service.process_frame(white, 0.2) is None
    assert service.process_frame(white, 0.4) is None
    outcome = service.process_frame(white, 0.8)

    assert outcome is not None
    assert outcome.user_id == "user_0000"
    assert outcome.counted is True
    assert identity_store.list_users() == ["user_0000"]
    assert beer_store.total_beers_by_user() == {"user_0000": 1}
    assert (tmp_path / "decisions.jsonl").exists()

    assert service.process_frame(black, 1.1) is None
    assert service.process_frame(white, 1.4) is None
    assert service.process_frame(white, 1.6) is None
    second = service.process_frame(white, 2.0)
    assert second is not None
    assert second.user_id == "user_0000"
    assert second.counted is False
    assert second.reason == "user_cooldown"
    assert beer_store.total_beers_by_user() == {"user_0000": 1}

    service.close()
    assert source.closed is True
