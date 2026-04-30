# preprocess_orl.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "ORL dataset"
IMG_SIZE = (64, 64)

def load_orl_dataset(dataset_path=DATASET_PATH):
    X = []
    y = []

    for subject_id in range(1, 41):  # s1 → s40
        subject_folder = os.path.join(dataset_path, f"s{subject_id}")

        for img_num in range(1, 11):  # 1.pgm → 10.pgm
            img_path = os.path.join(subject_folder, f"{img_num}.pgm")

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize
            img = cv2.resize(img, IMG_SIZE)

            # Normalize
            img = img.astype(np.float32) / 255.0

            # Flatten
            img_flat = img.flatten()

            X.append(img_flat)
            y.append(subject_id)

    X = np.array(X)
    y = np.array(y)

    return X, y


def prepare_data():
    X, y = load_orl_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    print("Dataset loaded:")
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_data()