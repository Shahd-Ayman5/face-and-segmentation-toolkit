import numpy as np
import pickle


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # 1. mean face
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. covariance trick
        cov = np.dot(X_centered, X_centered.T)

        # 3. eigen decomposition
        eigenvalues, eigenvectors_small = np.linalg.eigh(cov)

        # 4. sort
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors_small = eigenvectors_small[:, idx]

        # 5. eigenfaces
        eigenfaces = np.dot(X_centered.T, eigenvectors_small)

        # 6. normalize
        eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

        # 7. keep top components
        self.eigenfaces = eigenfaces[:, :self.n_components]

        # IMPORTANT: store training projections (for recognition)
        self.train_projections = np.dot(X_centered, self.eigenfaces)

        return self

    def transform(self, X):
        return np.dot(X - self.mean, self.eigenfaces)




    # SAVE MODEL
    def save(self, path="pca_model.pkl"):
        model = {
            "mean": self.mean,
            "eigenfaces": self.eigenfaces,
            "train_projections": self.train_projections,
            "n_components": self.n_components
        }

        with open(path, "wb") as f:
            pickle.dump(model, f)

    # LOAD MODEL
    @staticmethod
    def load(path="pca_model.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)

        pca = PCA(model["n_components"])
        pca.mean = model["mean"]
        pca.eigenfaces = model["eigenfaces"]
        pca.train_projections = model["train_projections"]

        return pca