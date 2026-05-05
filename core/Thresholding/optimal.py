import numpy as np


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _compute_class_means(gray: np.ndarray, T: float):
    """Return the mean intensity of pixels below and above threshold T."""
    below = gray[gray <= T]
    above = gray[gray > T]

    mu1 = float(below.mean()) if below.size > 0 else 0.0
    mu2 = float(above.mean()) if above.size > 0 else 255.0
    return mu1, mu2


def optimal_threshold(
    image: np.ndarray,
    max_iter: int = 100,
    tol: float = 0.5,
):
    
    if image.ndim != 2:
        raise ValueError("optimal_threshold expects a 2-D (grayscale) array.")

    gray = image.astype(np.float64)

    # Initialise T as the midpoint of the intensity range
    T = (float(gray.min()) + float(gray.max())) / 2.0
    history = [T]

    for i in range(1, max_iter + 1):
        mu1, mu2 = _compute_class_means(gray, T)
        T_new = (mu1 + mu2) / 2.0
        history.append(T_new)

        if abs(T_new - T) < tol:
            T = T_new
            break
        T = T_new
    else:
        i = max_iter  # ran to completion without converging

    threshold = int(round(T))
    binary_image = ((gray > T) * 255).astype(np.uint8)

    return threshold, binary_image, i, history


# ---------------------------------------------------------------------------
# Convenience wrapper (matches the interface expected by the GUI)
# ---------------------------------------------------------------------------

def apply_optimal_threshold(image: np.ndarray, **kwargs):
    threshold, binary, iterations, history = optimal_threshold(image, **kwargs)
    return {
        "binary": binary,
        "threshold": threshold,
        "iterations": iterations,
        "history": history,
    }


