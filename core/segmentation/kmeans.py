"""
K-Means Image Segmentation
===========================
Pure-NumPy implementation (no sklearn) so it integrates cleanly with the
rest of the project without extra heavy dependencies.

Supports both grayscale and RGB/BGR images.

Public API
----------
kmeans_segment(image, k=3, max_iter=100, tol=1e-4, n_init=3, random_state=None)
    -> (labels, segmented_image, centroids, inertia, iterations)

apply_kmeans_segmentation(image, k=3, **kwargs)
    -> dict with keys 'segmented', 'labels', 'centroids', 'inertia', 'iterations'
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core K-Means
# ---------------------------------------------------------------------------

def _init_centroids(pixels: np.ndarray, k: int, rng: np.random.Generator):
    """K-Means++ initialisation for better convergence."""
    n = pixels.shape[0]
    # Choose first centroid uniformly at random
    idx = rng.integers(0, n)
    centroids = [pixels[idx].astype(np.float64)]

    for _ in range(1, k):
        # Squared distances to the nearest existing centroid
        dists = np.array([
            min(np.sum((p - c) ** 2) for c in centroids)
            for p in pixels
        ])
        # Sample proportional to distance
        probs = dists / dists.sum()
        cumprobs = np.cumsum(probs)
        r = rng.random()
        new_idx = int(np.searchsorted(cumprobs, r))
        centroids.append(pixels[new_idx].astype(np.float64))

    return np.array(centroids)


def _assign_labels(pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return index of nearest centroid for every pixel (vectorised)."""
    # pixels:    (N, D)
    # centroids: (k, D)
    diff = pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]   # (N, k, D)
    sq_dists = np.sum(diff ** 2, axis=2)                              # (N, k)
    return np.argmin(sq_dists, axis=1)                                # (N,)


def _run_single(pixels: np.ndarray, k: int, max_iter: int, tol: float, rng):
    """One full K-Means run; returns (labels, centroids, inertia, iters)."""
    centroids = _init_centroids(pixels, k, rng)

    for it in range(1, max_iter + 1):
        labels = _assign_labels(pixels, centroids)

        new_centroids = np.array([
            pixels[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if shift < tol:
            break

    # Inertia: sum of squared distances to assigned centroid
    inertia = float(np.sum(
        np.sum((pixels - centroids[labels]) ** 2, axis=1)
    ))
    return labels, centroids, inertia, it


# ---------------------------------------------------------------------------
# Public segmentation function
# ---------------------------------------------------------------------------

def kmeans_segment(
    image: np.ndarray,
    k: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 3,
    random_state=None,
):
    """
    Segment *image* into *k* clusters using K-Means.

    Parameters
    ----------
    image        : H×W (grayscale) or H×W×3 (colour) uint8 numpy array.
    k            : Number of clusters / segments.
    max_iter     : Maximum iterations per run.
    tol          : Convergence tolerance on centroid shift.
    n_init       : Number of independent restarts; best result is kept.
    random_state : int or None – for reproducibility.

    Returns
    -------
    labels         : H×W int array with cluster index 0…k-1 per pixel.
    segmented      : H×W or H×W×3 uint8 image where each pixel is replaced
                     by its cluster's mean colour.
    centroids      : (k, D) float64 array of final cluster centres.
    inertia        : float – within-cluster sum of squared distances.
    iterations     : int   – iterations used in the best run.
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2-D (grayscale) or 3-D (colour).")

    is_color = image.ndim == 3
    h, w = image.shape[:2]

    # Flatten to (N, D) pixel matrix
    pixels = image.reshape(-1, image.shape[2]).astype(np.float64) if is_color \
        else image.reshape(-1, 1).astype(np.float64)

    rng = np.random.default_rng(random_state)

    best_labels, best_centroids, best_inertia, best_iters = None, None, np.inf, 0

    for _ in range(n_init):
        labels, centroids, inertia, iters = _run_single(pixels, k, max_iter, tol, rng)
        if inertia < best_inertia:
            best_labels, best_centroids, best_inertia, best_iters = \
                labels, centroids, inertia, iters

    # Build segmented image: replace each pixel with its centroid colour
    segmented_pixels = best_centroids[best_labels].astype(np.uint8)

    if is_color:
        segmented = segmented_pixels.reshape(h, w, image.shape[2])
    else:
        segmented = segmented_pixels.reshape(h, w)

    labels_2d = best_labels.reshape(h, w)

    return labels_2d, segmented, best_centroids, best_inertia, best_iters


# ---------------------------------------------------------------------------
# Convenience wrapper (matches GUI interface)
# ---------------------------------------------------------------------------

def apply_kmeans_segmentation(image: np.ndarray, k: int = 3, **kwargs):
    """
    Thin wrapper that returns a dict for easy GUI consumption.

    Returns
    -------
    dict with keys:
        'segmented'  : H×W or H×W×3 uint8 segmented image
        'labels'     : H×W int cluster-label map
        'centroids'  : (k, D) centroid array
        'inertia'    : float
        'iterations' : int
        'k'          : int (echo back the k used)
    """
    labels, segmented, centroids, inertia, iterations = kmeans_segment(
        image, k=k, **kwargs
    )
    return {
        "segmented": segmented,
        "labels": labels,
        "centroids": centroids,
        "inertia": inertia,
        "iterations": iterations,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- Grayscale test ---
    gray = np.zeros((64, 64), dtype=np.uint8)
    gray[:32, :32] = 50
    gray[:32, 32:] = 150
    gray[32:, :] = 220
    noise = rng.integers(-10, 11, gray.shape)
    gray = np.clip(gray.astype(int) + noise, 0, 255).astype(np.uint8)

    res = apply_kmeans_segmentation(gray, k=3, random_state=42)
    print("=== Grayscale K-Means ===")
    print(f"k={res['k']}, inertia={res['inertia']:.2f}, iters={res['iterations']}")
    print(f"Centroids: {res['centroids'].flatten().round(1)}")
    print(f"Unique labels: {np.unique(res['labels'])}")

    # --- Colour test ---
    color = np.zeros((64, 64, 3), dtype=np.uint8)
    color[:32, :32] = [200, 50, 50]    # reddish
    color[:32, 32:] = [50, 200, 50]    # greenish
    color[32:, :] = [50, 50, 200]      # bluish
    noise3 = rng.integers(-15, 16, color.shape)
    color = np.clip(color.astype(int) + noise3, 0, 255).astype(np.uint8)

    res2 = apply_kmeans_segmentation(color, k=3, random_state=7)
    print("\n=== Colour K-Means ===")
    print(f"k={res2['k']}, inertia={res2['inertia']:.2f}, iters={res2['iterations']}")
    print(f"Centroids:\n{res2['centroids'].round(1)}")
    print(f"Unique labels: {np.unique(res2['labels'])}")