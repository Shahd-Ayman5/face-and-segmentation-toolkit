from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from core.face.pca_model import PCA
from core.face.process_orl import prepare_data


@dataclass(frozen=True)
class MatchResult:
	index: int
	subject_id: int
	image_path: Path
	distance: float


class FaceClassifier:
	def __init__(self, model_path: Optional[str] = None):
		self._model_path = Path(model_path) if model_path else Path(__file__).resolve().parents[2] / "face-and-segmentation-toolkit/pca_model.pkl"
		self._pca: Optional[PCA] = None
		self._train_projections: Optional[np.ndarray] = None
		self._subjects: Optional[np.ndarray] = None
		self._paths = None

	def load(self) -> "FaceClassifier":
		if self._pca is not None and self._train_projections is not None and self._subjects is not None and self._paths is not None:
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
		distances = np.linalg.norm(self._train_projections - projection, axis=1)
		index = int(np.argmin(distances))

		return MatchResult(
			index=index,
			subject_id=int(self._subjects[index]),
			image_path=self._paths[index],
			distance=float(distances[index]),
		)
