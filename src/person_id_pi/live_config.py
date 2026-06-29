from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 15


@dataclass(frozen=True)
class MotionConfig:
    width: int = 320
    height: int = 180
    sample_fps: float = 5.0
    pixel_delta: int = 25
    changed_ratio: float = 0.02
    sustained_samples: int = 2


@dataclass(frozen=True)
class EventConfig:
    pre_roll_seconds: float = 2.0
    quiet_seconds: float = 3.0
    max_seconds: float = 30.0
    inference_stride: int = 3
    multi_person_frames: int = 2
    cooldown_seconds: float = 600.0
    object_min_seen_frames: int = 5
    object_iou_threshold: float = 0.3
    association_max_distance: float = 0.35


@dataclass(frozen=True)
class BeerConfig:
    model_path: str = "yolov8n.pt"
    confidence_min: float = 0.25
    can_aspect_ratio_min: float = 0.65
    can_aspect_ratio_max: float = 2.4


@dataclass(frozen=True)
class StorageConfig:
    identity_store: str = "profiles/face_templates.json"
    beer_store: str = "profiles/beverage_events.json"
    evidence_dir: str = "events"
    decision_log: str = "events/decisions.jsonl"
    record_evidence: bool = True


@dataclass(frozen=True)
class LiveConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    event: EventConfig = field(default_factory=EventConfig)
    beer: BeerConfig = field(default_factory=BeerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def load(cls, path: Path | str) -> "LiveConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Live configuration must be a JSON object")
        config = cls(
            camera=CameraConfig(**_section(payload, "camera")),
            motion=MotionConfig(**_section(payload, "motion")),
            event=EventConfig(**_section(payload, "event")),
            beer=BeerConfig(**_section(payload, "beer")),
            storage=StorageConfig(**_section(payload, "storage")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if min(self.camera.width, self.camera.height, self.camera.fps) <= 0:
            raise ValueError("Camera dimensions and FPS must be positive")
        if self.motion.sample_fps <= 0 or self.motion.sample_fps > self.camera.fps:
            raise ValueError("motion.sample_fps must be between 0 and camera FPS")
        if not 0.0 <= self.motion.changed_ratio <= 1.0:
            raise ValueError("motion.changed_ratio must be between 0 and 1")
        if self.event.inference_stride < 1:
            raise ValueError("event.inference_stride must be at least 1")
        if self.event.quiet_seconds <= 0 or self.event.max_seconds <= 0:
            raise ValueError("Event timeouts must be positive")
        if self.event.object_min_seen_frames < 1:
            raise ValueError("event.object_min_seen_frames must be at least 1")
        if not 0.0 <= self.event.association_max_distance <= 1.0:
            raise ValueError("event.association_max_distance must be between 0 and 1")


def _section(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {name!r} must be an object")
    return value
