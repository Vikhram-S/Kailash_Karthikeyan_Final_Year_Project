import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class FaceBox:
    box: Tuple[int, int, int, int]  # x, y, w, h
    score: float


class FaceDetector:
    """Face detector using OpenCV Haar cascades (no mediapipe dependency)."""

    def __init__(self, min_confidence: float = 0.6, model_selection: int = 1):
        self.min_confidence = min_confidence
        # Use bundled frontal-face Haar cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image_bgr: np.ndarray) -> List[FaceBox]:
        """Run face detection on a BGR image using Haar cascades."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        boxes: List[FaceBox] = []
        # Haar cascades don't give a probability, so we just report 1.0
        for (x, y, w, h) in faces:
            boxes.append(FaceBox(box=(x, y, w, h), score=1.0))

        return boxes

    @staticmethod
    def draw_detections(
        image_bgr: np.ndarray,
        boxes: List[FaceBox],
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw detected face boxes with an optional label (e.g. person's name)."""
        out = image_bgr.copy()
        for idx, fb in enumerate(boxes, start=1):
            x, y, w, h = fb.box
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = label if label else f"Face {idx}"
            cv2.putText(
                out,
                text,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return out

