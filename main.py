import sys
import shutil
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

from controllers.main_controller import AppController

def test_pca():
    from core.face.process_orl import prepare_data
    from core.face.pca_model import PCA
    import numpy as np
    import shutil
    from pathlib import Path

    model_path = Path(__file__).parent / "pca_model.pkl"

    print("=" * 40)
    print("First run: Computing Eigenfaces...")
    print("=" * 40)

    X_train, X_test, y_train, y_test, train_paths, test_paths = prepare_data()

    print(f"Train shape : {X_train.shape}")
    print(f"Test shape  : {X_test.shape}")

    # =========================
    # Train PCA
    # =========================
    pca = PCA(n_components=50)
    pca.fit(X_train)

    print(f"Eigenfaces  : {pca.eigenfaces.shape}")
    print(f"Mean face   : {pca.mean.shape}")
    print(f"Train proj  : {pca.train_projections.shape}")

    # =========================
    # Train Accuracy
    # =========================
    correct_train = 0

    for i in range(len(pca.train_projections)):
        dists = np.linalg.norm(
            pca.train_projections - pca.train_projections[i],
            axis=1
        )

        dists[i] = np.inf

        predicted = y_train[int(np.argmin(dists))]

        if predicted == y_train[i]:
            correct_train += 1

    train_acc = correct_train / len(y_train) * 100

    print(f"\nTrain Accuracy : {correct_train}/{len(y_train)} = {train_acc:.1f}%")

    # =========================
    # Test Accuracy
    # =========================
    proj_test = pca.transform(X_test)

    correct_test = 0

    for i in range(len(proj_test)):
        dists = np.linalg.norm(
            pca.train_projections - proj_test[i],
            axis=1
        )

        predicted = y_train[int(np.argmin(dists))]

        if predicted == y_test[i]:
            correct_test += 1

    test_acc = correct_test / len(proj_test) * 100

    print(f"Test Accuracy  : {correct_test}/{len(proj_test)} = {test_acc:.1f}%")

    # =========================
    # Simple Evaluation Message
    # =========================
    if train_acc > 95 and test_acc < 75:
        print("⚠️ Possible Overfitting")
    elif train_acc < 75 and test_acc < 75:
        print("⚠️ Possible Underfitting")
    else:
        print("✅ Model looks reasonable")

    # =========================
    # Save model
    # =========================
    pca.save(str(model_path))
    print(f"✅ Model saved to: {model_path}")

    # =========================
    # Save test images (for UI demo)
    # =========================
    test_dir = Path(__file__).parent / "copytest_images"

    if not test_dir.exists():
        test_dir.mkdir()

        for path in test_paths:
            subject_id = path.parent.name
            new_name = f"{subject_id}_{path.name}"
            shutil.copy(path, test_dir / new_name)

        print(f"✅ Test images saved to: {test_dir}")

    print("=" * 40)

def main():

    # visualize_eigenfaces()
    # test_pca()

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Load style
    qss_path = Path(__file__).parent / "ui" / "dark.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text())

    window = QMainWindow()
    AppController(window)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
        main()




#________________________________________________________________________





def visualize_eigenfaces():
    from core.face.pca_model import PCA
    import numpy as np
    import cv2

    model_path = Path(__file__).parent / "face-and-segmentation-toolkit/pca_model.pkl"
    pca = PCA.load(str(model_path))

    # show mean face
    mean_img = pca.mean.reshape(64, 64)
    mean_img = (mean_img * 255).astype(np.uint8)
    cv2.imshow("Mean Face", cv2.resize(mean_img, (200, 200)))

    # show top 10 eigenfaces
    for i in range(20):
        ef = pca.eigenfaces[:, i].reshape(64, 64)

        # normalize to 0-255 for display
        ef = (ef - ef.min()) / (ef.max() - ef.min()) * 255
        ef = ef.astype(np.uint8)

        cv2.imshow(f"Eigenface {i+1}", cv2.resize(ef, (200, 200)))

    print("Showing eigenfaces — press any key to close all")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
