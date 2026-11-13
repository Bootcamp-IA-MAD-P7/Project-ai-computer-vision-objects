"""Reusable detector wrapper around the trained YOLO11n model."""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.config import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_OUTPUT,
    DEFAULT_WEIGHTS,
)


class AeroVisionDetector:
    """Load the trained weights once and run detections on frames or files."""

    def __init__(
        self,
        weights: Path = DEFAULT_WEIGHTS,
        conf: float = DEFAULT_CONF,
        imgsz: int = DEFAULT_IMGSZ,
        device: str | None = None,
    ) -> None:
        # Imported here so that argument errors are reported without loading PyTorch.
        from ultralytics import YOLO

        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(str(weights))

    @staticmethod
    def count_classes(result) -> Counter:
        """Count how many detections of each class one result contains."""
        return Counter(
            result.names[int(class_id)] for class_id in result.boxes.cls.tolist()
        )

    def predict_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Counter]:
        """Detect objects in one frame; return the annotated frame and class counts."""
        result = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        return result.plot(), self.count_classes(result)

    def predict_source(self, source: str | Path, output: str | Path = DEFAULT_OUTPUT) -> Iterator:
        """Yield one result per image or video frame, saving the annotated output."""
        yield from self.model.predict(
            source=str(source),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            save=True,
            stream=True,
            verbose=False,
            project=str(output),
            name="predictions",
            exist_ok=True,
        )
