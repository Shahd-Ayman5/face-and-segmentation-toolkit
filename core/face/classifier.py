from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from collections import Counter

from core.face.pca_model import PCA
from core.face.process_orl import prepare_data


@dataclass(frozen=True)
class MatchResult:
    indices: list[int]
    subject_id: int
    image_path: Path
    distance: float


class FaceClassifier:
    def __init__(self, model_path: Optional[str] = None, k: int = 5):
        self._model_path = (
            Path(model_path)
            if model_path
            else Path(__file__).resolve().parents[2]
            / "face-and-segmentation-toolkit/pca_model.pkl"
        )

        self.k = k
        self._pca: Optional[PCA] = None
        self._train_projections: Optional[np.ndarray] = None
        self._subjects: Optional[np.ndarray] = None
        self._paths = None

    def load(self) -> "FaceClassifier":
        if (
            self._pca is not None
            and self._train_projections is not None
            and self._subjects is not None
            and self._paths is not None
        ):
            return self

        self._pca = PCA.load(str(self._model_path))
        self._train_projections = self._pca.train_projections

        _, _, y_train, _, train_paths, _ = prepare_data()
        self._subjects = y_train
        self._paths = train_paths
        return self

    def predict(self, face_vector: np.ndarray) -> MatchResult:
        self.load()

        if face_vector.ndim == 1:
            face_vector = np.expand_dims(face_vector, axis=0)

        projection = self._pca.transform(face_vector)

        distances = np.linalg.norm(
            self._train_projections - projection,
            axis=1
        )

        # indices of k nearest neighbors
        knn_indices = np.argsort(distances)[: self.k]

        # labels of nearest neighbors
        knn_labels = self._subjects[knn_indices]

        # majority voting
        predicted_subject = Counter(knn_labels).most_common(1)[0][0]

        # choose closest sample among k for path/display
        best_index = knn_indices[0]

        return MatchResult(
            indices=knn_indices.tolist(),
            subject_id=int(predicted_subject),
            image_path=self._paths[best_index],
            distance=float(distances[best_index]),
        )