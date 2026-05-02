import sys
import shutil
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

from controllers.main_controller import AppController


def main():

    # visualize_eigenfaces()

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
