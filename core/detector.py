"""
core/detector.py

Ball detection using a YOLO model (Ultralytics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single ball detection result for one video frame."""
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float

    @property
    def radius(self) -> float:
        """Estimated radius based on the bounding box."""
        x_min, y_min, x_max, y_max = self.bbox
        return max(x_max - x_min, y_max - y_min) / 2.0


class BallDetector:
    """Detects the foosball in individual video frames using YOLO."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        class_id: Optional[int] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        # COCO "sports ball" is 32. If fine-tuned model has 1 class (ball), it's 0.
        self.class_id = class_id if class_id is not None else 32

        resolved_model = model_path or "yolov8n.pt"
        logger.info("Loading YOLO model from: %s", resolved_model)
        self._model = self._load_model(resolved_model)

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO  # type: ignore
            return YOLO(model_path)
        except ImportError as exc:
            raise ImportError("Install ultralytics: pip install ultralytics") from exc

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        results = self._model(frame, verbose=False)
        best: Optional[Detection] = None

        for result in results:
            for box in result.boxes:
                if self.class_id is not None and int(box.cls[0]) != self.class_id:
                    continue

                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                x_min, y_min, x_max, y_max = (int(v) for v in box.xyxy[0])
                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2

                detection = Detection(
                    bbox=(x_min, y_min, x_max, y_max),
                    center=(cx, cy),
                    confidence=conf,
                )

                if best is None or conf > best.confidence:
                    best = detection

        return best