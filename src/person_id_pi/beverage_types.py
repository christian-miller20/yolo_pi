from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

# Canonical beverage labels used throughout detection, events, and storage.
BeverageLabel = Literal["cup", "can", "bottle"]
BeerLabels = {"can", "bottle", "cup"}


@dataclass(frozen=True)
class BeverageDetection:
    # (x1, y1, x2, y2) bounding box in image coordinates.
    bbox: Tuple[int, int, int, int]
    label: BeverageLabel
    score: float  # Confidence score from the beverage detector (e.g., YOLOv8)


@dataclass(frozen=True)
class BeverageEvent:
    # Deterministic id for idempotent persistence.
    event_id: str
    # Logical video identifier (usually source path or normalized stem).
    video_id: str
    # Frame index where the event was first observed.
    frame_idx: int
    # Person track id from face pipeline.
    track_id: int
    # Resolved accepted user id.
    user_id: str
    # Beverage class.
    beverage_label: BeverageLabel
    # Object tracker id for distinct-event policy.
    object_track_id: int
    # Confidence for this event (typically detection score at first observation).
    confidence: float
    # UTC timestamp string (ISO-8601).
    timestamp_utc: str
    # Audit artifact and normalized category added by the live service.
    evidence_path: Optional[str] = None
    container_category: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_id": self.event_id,
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "user_id": self.user_id,
            "beverage_label": self.beverage_label,
            "object_track_id": self.object_track_id,
            "confidence": self.confidence,
            "timestamp_utc": self.timestamp_utc,
            "evidence_path": self.evidence_path,
            "container_category": self.container_category,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BeverageEvent":
        return cls(
            event_id=str(payload["event_id"]),
            video_id=str(payload["video_id"]),
            frame_idx=int(payload["frame_idx"]),
            track_id=int(payload["track_id"]),
            user_id=str(payload["user_id"]),
            beverage_label=str(payload["beverage_label"]),  # type: ignore[arg-type]
            object_track_id=int(payload["object_track_id"]),
            confidence=float(payload["confidence"]),
            timestamp_utc=str(payload["timestamp_utc"]),
            evidence_path=(
                str(payload["evidence_path"])
                if payload.get("evidence_path") is not None
                else None
            ),
            container_category=(
                str(payload["container_category"])
                if payload.get("container_category") is not None
                else None
            ),
            session_id=(
                str(payload["session_id"])
                if payload.get("session_id") is not None
                else None
            ),
        )
