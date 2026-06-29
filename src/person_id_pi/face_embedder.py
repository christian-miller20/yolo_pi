from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .face_types import FaceDetection, FaceEmbedding


@dataclass
class FaceEmbedderConfig:
    min_face_size: int = 80
    model_name: str = "buffalo_l"
    providers: Sequence[str] = ("CPUExecutionProvider",)
    blur_ref: float = 100.0
    size_ref: int = 160


class FaceEmbedder:
    """InsightFace detection and ArcFace embedding adapter."""

    def __init__(self, config: Optional[FaceEmbedderConfig] = None) -> None:
        self.config = config or FaceEmbedderConfig()
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required. Install with: pip install insightface onnxruntime"
            ) from exc

        self._app = FaceAnalysis(
            name=self.config.model_name,
            providers=list(self.config.providers),
            allowed_modules=["detection", "recognition"],
        )
        # ctx_id=-1 uses CPU; det_size controls the detector input size.
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    def _blur_score(self, crop: np.ndarray) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(1.0, variance / max(1.0, self.config.blur_ref)))

    def _size_score(self, width: int, height: int) -> float:
        size = min(width, height)
        return float(min(1.0, size / max(1, self.config.size_ref)))

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        faces = self._app.get(frame)
        detections: List[FaceDetection] = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            # Reject tiny faces; embeddings are unstable at very small sizes.
            if min(w, h) < self.config.min_face_size:
                continue
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2].copy()
            # Keep landmarks if InsightFace returns them (used for alignment later).
            kps = None
            if hasattr(face, "kps") and face.kps is not None:
                kps = [(float(x), float(y)) for x, y in face.kps.tolist()]
            # Detector confidence is used as a light-weight "quality" signal.
            det_score = float(getattr(face, "det_score", 0.0))
            size_score = self._size_score(w, h)
            blur_score = self._blur_score(crop)
            quality = det_score * size_score * blur_score
            detections.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    landmarks=kps,
                    quality=quality,
                    det_score=det_score,
                    size_score=size_score,
                    blur_score=blur_score,
                    aligned_crop=crop,
                    embedding=getattr(face, "embedding", None),
                )
            )
        return detections

    def embed(self, face: FaceDetection) -> FaceEmbedding:
        # InsightFace returns the embedding on the detected face object.
        if face.embedding is None:
            raise RuntimeError(
                "No embedding found on detection. Ensure InsightFace is configured."
            )
        embedding = np.asarray(face.embedding, dtype=np.float32)
        return FaceEmbedding(
            embedding=embedding,
            quality=face.quality,
            bbox=face.bbox,
            landmarks=face.landmarks,
        )
