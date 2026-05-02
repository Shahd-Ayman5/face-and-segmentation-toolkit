import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "ORL dataset"
IMG_SIZE = (64, 64)


def load_orl_dataset(dataset_path=DATASET_PATH):
    X, y, paths = [], [], []
    for subject_id in range(1, 41):
        for img_num in range(1, 11):
            img_path = Path(dataset_path) / f"s{subject_id}" / f"{img_num}.pgm"
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0
            X.append(img.flatten())
            y.append(subject_id)
            paths.append(img_path)
    return np.array(X), np.array(y), paths



def prepare_data():
    X, y, paths = load_orl_dataset()
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.3, stratify=y, random_state=42
    )
    return (
        X[idx_train], X[idx_test],
        y[idx_train], y[idx_test],
        [paths[i] for i in idx_train],
        [paths[i] for i in idx_test],
    )