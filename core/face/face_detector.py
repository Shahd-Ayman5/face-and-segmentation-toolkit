from __future__ import annotations

import time
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FaceDetectionResult:
    faces: list
    visualisation: np.ndarray
    computation_time_ms: float
    num_faces: int


def detect_faces(
        image: np.ndarray,
        scale_factor: float = 1.3,
        min_neighbors: int = 5,
) -> FaceDetectionResult:

    start = time.perf_counter()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    vis = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    elapsed = (time.perf_counter() - start) * 1000

    return FaceDetectionResult(
        faces=faces,
        visualisation=vis,
        computation_time_ms=elapsed,
        num_faces=len(faces)
    )