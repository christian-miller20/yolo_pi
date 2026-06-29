from __future__ import annotations

import json
import signal
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .beer_detector import BeerDetection, BeerDetector
from .beverage_store import BeverageStore
from .beverage_types import BeverageEvent, BeverageLabel
from .camera import FrameSource
from .face_embedder import FaceEmbedder
from .face_types import FaceDetection, FaceEmbedding, IdentityDecision
from .identity_engine import IdentityEngine
from .live_config import LiveConfig


class MotionDetector:
    def __init__(
        self,
        width: int,
        height: int,
        pixel_delta: int,
        changed_ratio: float,
        sustained_samples: int,
    ) -> None:
        self.width = width
        self.height = height
        self.pixel_delta = pixel_delta
        self.changed_ratio = changed_ratio
        self.sustained_samples = max(1, sustained_samples)
        self._previous: Optional[np.ndarray] = None
        self._consecutive = 0

    def update(self, frame: np.ndarray) -> Tuple[bool, bool, float]:
        small = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._previous is None:
            self._previous = gray
            return False, False, 0.0
        delta = cv2.absdiff(self._previous, gray)
        self._previous = gray
        ratio = float(np.count_nonzero(delta >= self.pixel_delta) / delta.size)
        moving = ratio >= self.changed_ratio
        self._consecutive = self._consecutive + 1 if moving else 0
        return moving, self._consecutive >= self.sustained_samples, ratio


@dataclass
class TrackedBeer:
    track_id: int
    category: BeverageLabel
    bbox: Tuple[int, int, int, int]
    score: float
    seen_count: int
    last_inference_index: int
    last_frame_index: int
    associated_seen_count: int


@dataclass
class ActiveEvent:
    session_id: str
    started_at: float
    last_motion_at: float
    raw_path: Optional[Path]
    writer: Optional[cv2.VideoWriter]
    frame_index: int = -1
    inference_index: int = 0
    faces: List[FaceEmbedding] = field(default_factory=list)
    multi_person_frames: int = 0
    face_boxes: Dict[int, List[Tuple[int, int, int, int]]] = field(default_factory=dict)
    beer_boxes: Dict[int, List[BeerDetection]] = field(default_factory=dict)
    beer_tracks: Dict[int, TrackedBeer] = field(default_factory=dict)
    next_beer_track_id: int = 1
    baseline_beers: List[BeerDetection] = field(default_factory=list)


@dataclass(frozen=True)
class EventOutcome:
    session_id: str
    accepted_identity: bool
    user_id: Optional[str]
    counted: bool
    reason: str
    evidence_path: Optional[str]


class LiveService:
    def __init__(
        self,
        config: LiveConfig,
        source: FrameSource,
        face_embedder: FaceEmbedder,
        identity: IdentityEngine,
        beer_detector: BeerDetector,
        beer_store: BeverageStore,
        log: Callable[[str], None] = print,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        async_inference: bool = True,
    ) -> None:
        self.config = config
        self.source = source
        self.face_embedder = face_embedder
        self.identity = identity
        self.beer_detector = beer_detector
        self.beer_store = beer_store
        self.log = log
        self.monotonic = monotonic
        self.utcnow = utcnow
        motion = config.motion
        self.motion = MotionDetector(
            width=motion.width,
            height=motion.height,
            pixel_delta=motion.pixel_delta,
            changed_ratio=motion.changed_ratio,
            sustained_samples=motion.sustained_samples,
        )
        pre_roll_frames = max(
            1, int(round(config.event.pre_roll_seconds * config.camera.fps))
        )
        self._pre_roll: Deque[Tuple[float, np.ndarray]] = deque(maxlen=pre_roll_frames)
        self._active: Optional[ActiveEvent] = None
        self._last_motion_sample_at = float("-inf")
        self._stop = False
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1) if async_inference else None
        self._pending: Optional[
            Tuple[int, Future[Tuple[List[FaceDetection], List[BeerDetection]]]]
        ] = None

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self.log("live_service_started")
        try:
            while not self._stop:
                ok, frame = self.source.read()
                if not ok or frame is None:
                    self.log("camera_frame_unavailable")
                    time.sleep(0.05)
                    continue
                outcome = self.process_frame(frame, self.monotonic())
                if outcome is not None:
                    self.log(
                        "event_finalized "
                        f"session={outcome.session_id} user={outcome.user_id} "
                        f"counted={outcome.counted} reason={outcome.reason}"
                    )
        finally:
            self.close()

    def process_frame(
        self, frame: np.ndarray, timestamp: Optional[float] = None
    ) -> Optional[EventOutcome]:
        now = self.monotonic() if timestamp is None else timestamp
        self._collect_inference(wait=False)
        sample_interval = 1.0 / self.config.motion.sample_fps
        moving = False
        triggered = False
        if now - self._last_motion_sample_at >= sample_interval:
            moving, triggered, _ = self.motion.update(frame)
            self._last_motion_sample_at = now

        if self._active is None:
            self._pre_roll.append((now, frame.copy()))
            if not triggered:
                return None
            self._start_event(now)
            active = self._require_active()
            if moving:
                active.last_motion_at = now
            self._schedule_inference(frame)
            return None

        active = self._require_active()
        active.frame_index += 1
        if active.writer is not None:
            active.writer.write(frame)
        if moving:
            active.last_motion_at = now
        if active.frame_index % self.config.event.inference_stride == 0:
            self._schedule_inference(frame)

        quiet = now - active.last_motion_at >= self.config.event.quiet_seconds
        expired = now - active.started_at >= self.config.event.max_seconds
        if quiet or expired:
            return self._finalize_event("quiet" if quiet else "max_duration")
        return None

    def close(self) -> Optional[EventOutcome]:
        if self._closed:
            return None
        outcome = self._finalize_event("shutdown") if self._active else None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self.source.close()
        self._closed = True
        self.log("live_service_stopped")
        return outcome

    def _start_event(self, now: float) -> None:
        stamp = self.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
        session_id = f"event-{stamp}"
        raw_path: Optional[Path] = None
        writer: Optional[cv2.VideoWriter] = None
        if self.config.storage.record_evidence:
            evidence_dir = Path(self.config.storage.evidence_dir)
            evidence_dir.mkdir(parents=True, exist_ok=True)
            raw_path = evidence_dir / f".{session_id}.raw.mp4"
            writer = cv2.VideoWriter(
                str(raw_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.config.camera.fps),
                (self.config.camera.width, self.config.camera.height),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"Unable to create event clip {raw_path}")
        active = ActiveEvent(
            session_id=session_id,
            started_at=now,
            last_motion_at=now,
            raw_path=raw_path,
            writer=writer,
        )
        if self._pre_roll:
            active.baseline_beers = self.beer_detector.detect(self._pre_roll[0][1])
        for _, buffered_frame in self._pre_roll:
            active.frame_index += 1
            if writer is not None:
                writer.write(buffered_frame)
        self._pre_roll.clear()
        self._active = active
        self.log(f"event_started session={session_id}")

    def _schedule_inference(self, frame: np.ndarray) -> None:
        active = self._require_active()
        if self._executor is None:
            faces, beer_detections = self._infer(frame)
            self._apply_inference(active.frame_index, faces, beer_detections)
            return
        if self._pending is not None:
            return
        future = self._executor.submit(self._infer, frame.copy())
        self._pending = (active.frame_index, future)

    def _infer(
        self, frame: np.ndarray
    ) -> Tuple[List[FaceDetection], List[BeerDetection]]:
        faces = self.face_embedder.detect(frame)
        return faces, self.beer_detector.detect(frame)

    def _collect_inference(self, wait: bool) -> None:
        if self._pending is None:
            return
        frame_index, future = self._pending
        if not wait and not future.done():
            return
        faces, beer_detections = future.result()
        self._pending = None
        self._apply_inference(frame_index, faces, beer_detections)

    def _apply_inference(
        self,
        frame_index: int,
        faces: List[FaceDetection],
        beer_detections: List[BeerDetection],
    ) -> None:
        active = self._require_active()
        if len(faces) > 1:
            active.multi_person_frames += 1
        if len(faces) == 1:
            active.faces.append(self.face_embedder.embed(faces[0]))
        active.face_boxes[frame_index] = [face.bbox for face in faces]

        active.beer_boxes[frame_index] = beer_detections
        self._update_beer_tracks(active, beer_detections, frame_index)
        active.inference_index += 1

    def _update_beer_tracks(
        self,
        active: ActiveEvent,
        detections: List[BeerDetection],
        frame_index: int,
    ) -> None:
        used: set[int] = set()
        for detection in detections:
            associated = self._bbox_is_near_face(active, detection.bbox, frame_index)
            best_id: Optional[int] = None
            best_iou = 0.0
            for track_id, track in active.beer_tracks.items():
                if track_id in used or track.category != detection.category:
                    continue
                if active.inference_index - track.last_inference_index > 3:
                    continue
                iou = _iou(track.bbox, detection.bbox)
                if iou >= self.config.event.object_iou_threshold and iou > best_iou:
                    best_id, best_iou = track_id, iou
            if best_id is None:
                best_id = active.next_beer_track_id
                active.next_beer_track_id += 1
                active.beer_tracks[best_id] = TrackedBeer(
                    track_id=best_id,
                    category=detection.category,
                    bbox=detection.bbox,
                    score=detection.score,
                    seen_count=1,
                    last_inference_index=active.inference_index,
                    last_frame_index=frame_index,
                    associated_seen_count=int(associated),
                )
            else:
                track = active.beer_tracks[best_id]
                track.bbox = detection.bbox
                track.score = max(track.score, detection.score)
                track.seen_count += 1
                track.last_inference_index = active.inference_index
                track.last_frame_index = frame_index
                track.associated_seen_count += int(associated)
            used.add(best_id)

    def _finalize_event(self, end_reason: str) -> EventOutcome:
        active = self._require_active()
        self._collect_inference(wait=True)
        if active.writer is not None:
            active.writer.release()
        identity_decision, identity_reason = self._resolve_identity(active)
        user_id = identity_decision.user_id if identity_decision.accepted else None
        candidate = self._best_beer_candidate(active)
        counted = False
        reason = identity_reason
        now_utc = self.utcnow()

        if user_id and candidate is None:
            reason = "no_qualified_container"
        elif user_id and candidate is not None:
            last = self.beer_store.latest_beer_timestamp(user_id)
            if last is not None:
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                within_cooldown = (
                    now_utc - last
                ).total_seconds() < self.config.event.cooldown_seconds
            else:
                within_cooldown = False
            if within_cooldown:
                reason = "user_cooldown"
            else:
                counted = True
                reason = "counted"

        evidence_path = self._finish_evidence(active, user_id, counted, reason)
        if counted and user_id and candidate is not None:
            event_id = sha1(
                f"{active.session_id}|{user_id}|{candidate.track_id}".encode("utf-8")
            ).hexdigest()[:16]
            self.beer_store.add_event(
                BeverageEvent(
                    event_id=event_id,
                    video_id=active.session_id,
                    frame_idx=0,
                    track_id=1,
                    user_id=user_id,
                    beverage_label=candidate.category,
                    object_track_id=candidate.track_id,
                    confidence=candidate.score,
                    timestamp_utc=now_utc.isoformat(),
                    evidence_path=evidence_path,
                    container_category=(
                        "cup" if candidate.category == "cup" else "can"
                    ),
                    session_id=active.session_id,
                )
            )
        outcome = EventOutcome(
            session_id=active.session_id,
            accepted_identity=bool(user_id),
            user_id=user_id,
            counted=counted,
            reason=reason,
            evidence_path=evidence_path,
        )
        self._append_decision(outcome, end_reason)
        self._active = None
        return outcome

    def _resolve_identity(self, active: ActiveEvent) -> Tuple[IdentityDecision, str]:
        if active.multi_person_frames >= self.config.event.multi_person_frames:
            return _rejected("multiple_people"), "multiple_people"
        tracklet = self.identity.aggregate_tracklet(active.faces)
        decision = self.identity.match(tracklet)
        if decision.accepted:
            if decision.user_id and self.identity.should_update_templates(
                decision, tracklet
            ):
                self.identity.update_templates(decision.user_id, tracklet)
            return decision, "identity_matched"
        block_reason = self.identity.auto_enroll_block_reason(decision, tracklet)
        if block_reason is not None:
            return decision, f"identity_{block_reason}"
        user_id = self.identity.store.generate_new_user_id()
        if not self.identity.update_templates(user_id, tracklet):
            return decision, "identity_enrollment_failed"
        return (
            IdentityDecision(
                user_id=user_id,
                score=decision.score,
                margin=decision.margin,
                accepted=True,
                reason="auto_enrolled_unknown",
            ),
            "identity_auto_enrolled",
        )

    def _best_beer_candidate(self, active: ActiveEvent) -> Optional[TrackedBeer]:
        candidates = [
            track
            for track in active.beer_tracks.values()
            if track.seen_count >= self.config.event.object_min_seen_frames
            and not self._was_present_before_event(active, track)
            and self._is_near_face(active, track)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.seen_count, item.score))

    def _was_present_before_event(
        self, active: ActiveEvent, track: TrackedBeer
    ) -> bool:
        return any(
            baseline.category == track.category
            and _iou(baseline.bbox, track.bbox)
            >= self.config.event.object_iou_threshold
            for baseline in active.baseline_beers
        )

    def _is_near_face(self, active: ActiveEvent, track: TrackedBeer) -> bool:
        del active
        return track.associated_seen_count > 0

    def _bbox_is_near_face(
        self,
        active: ActiveEvent,
        bbox: Tuple[int, int, int, int],
        frame_index: int,
    ) -> bool:
        face_boxes = active.face_boxes.get(frame_index, [])
        if len(face_boxes) != 1:
            return False
        face_center = _center(face_boxes[0])
        beer_center = _center(bbox)
        distance = np.hypot(
            face_center[0] - beer_center[0], face_center[1] - beer_center[1]
        )
        diagonal = np.hypot(self.config.camera.width, self.config.camera.height)
        return float(distance / max(1.0, diagonal)) <= (
            self.config.event.association_max_distance
        )

    def _finish_evidence(
        self,
        active: ActiveEvent,
        user_id: Optional[str],
        counted: bool,
        reason: str,
    ) -> Optional[str]:
        if active.raw_path is None:
            return None
        display_label = (
            self.identity.store.display_label(user_id) if user_id else "unknown"
        )
        suffix = "counted" if counted else "rejected"
        output = active.raw_path.parent / f"{active.session_id}.{suffix}.mp4"
        cap = cv2.VideoCapture(str(active.raw_path))
        writer: Optional[cv2.VideoWriter] = None
        frame_idx = 0
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or self.config.camera.fps)
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Unable to annotate event clip {output}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                for bbox in active.face_boxes.get(frame_idx, []):
                    _draw_box(frame, bbox, display_label, (0, 200, 0))
                for detection in active.beer_boxes.get(frame_idx, []):
                    label = "CUP" if detection.category == "cup" else "CAN"
                    _draw_box(frame, detection.bbox, label, (0, 255, 255))
                cv2.putText(
                    frame,
                    reason,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if counted else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        active.raw_path.unlink(missing_ok=True)
        return str(output)

    def _append_decision(self, outcome: EventOutcome, end_reason: str) -> None:
        path = Path(self.config.storage.decision_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": outcome.session_id,
            "timestamp_utc": self.utcnow().isoformat(),
            "user_id": outcome.user_id,
            "display_name": (
                self.identity.store.get_display_name(outcome.user_id)
                if outcome.user_id
                else None
            ),
            "accepted_identity": outcome.accepted_identity,
            "counted": outcome.counted,
            "reason": outcome.reason,
            "end_reason": end_reason,
            "evidence_path": outcome.evidence_path,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()

    def _require_active(self) -> ActiveEvent:
        if self._active is None:
            raise RuntimeError("No active event")
        return self._active


def install_signal_handlers(service: LiveService) -> None:
    def _request_stop(signum: int, frame: object) -> None:
        del signum, frame
        service.request_stop()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return float(intersection / union) if union > 0 else 0.0


def _draw_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _rejected(reason: str) -> IdentityDecision:
    return IdentityDecision(
        user_id=None,
        score=0.0,
        margin=0.0,
        accepted=False,
        reason=reason,
    )
