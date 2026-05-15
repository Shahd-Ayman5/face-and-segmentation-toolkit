from __future__ import annotations
import numpy as np
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def spectral_threshold(image: np.ndarray, n_modes: int = 3) -> tuple[np.ndarray, list[int]]:
    
    if image.ndim != 2:
        raise ValueError("spectral_threshold expects a 2-D grayscale image.")

    n_thresholds = n_modes - 1

    if n_thresholds < 1:
        raise ValueError("n_modes must be at least 2.")

    # Build histogram once
    hist = _histogram(image)
    global_mean = _global_mean(hist)

    # Find optimal thresholds
    thresholds = _find_optimal_thresholds(hist, global_mean, n_thresholds)

    # Build output image
    result = _apply_thresholds(image, thresholds, n_modes)

    return result, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _histogram(image: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    return hist.astype(np.float64)


def _global_mean(hist: np.ndarray) -> float:
    intensities = np.arange(256, dtype=np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    return float(np.dot(intensities, hist) / total)





def _find_optimal_thresholds(hist: np.ndarray,
                             global_mean: float,
                             n_thresholds: int) -> list[int]:

    candidates = list(range(1, 255))          # valid threshold range: 1–254

    best_var = -1.0
    best_thresholds: list[int] = [64 * i for i in range(1, n_thresholds + 1)]

    # Pre-compute cumulative sums for fast region statistics
    cumsum      = np.cumsum(hist)             # cumulative pixel count
    cumsum_w    = np.cumsum(np.arange(256, dtype=np.float64) * hist)  # cumulative weighted count
    total       = cumsum[-1]

    for combo in combinations(candidates, n_thresholds):
        # combo is always in ascending order (combinations preserves order)
        boundaries = [0] + list(combo) + [256]
        n_classes  = len(boundaries) - 1

        var = 0.0
        for i in range(n_classes):
            lo, hi = boundaries[i], boundaries[i + 1] - 1   # inclusive hi for cumsum indexing
            # pixel count in [lo, hi]
            w = cumsum[hi] - (cumsum[lo - 1] if lo > 0 else 0)
            if w == 0:
                continue
            # weighted sum in [lo, hi]
            ws = cumsum_w[hi] - (cumsum_w[lo - 1] if lo > 0 else 0)
            mean = ws / w
            var += (w / total) * (mean - global_mean) ** 2

        if var > best_var:
            best_var = var
            best_thresholds = list(combo)

    return best_thresholds


def _apply_thresholds(image: np.ndarray,
                      thresholds: list[int],
                      n_modes: int) -> np.ndarray:
    
    result = np.zeros_like(image, dtype=np.uint8)
    boundaries = [0] + thresholds + [256]

    # Evenly spaced output grey values: 0, 255/(n-1), 2*255/(n-1), ..., 255
    grey_levels = [int(round(255 * i / (n_modes - 1))) for i in range(n_modes)]

    for i in range(n_modes):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        mask = (image >= lo) & (image < hi)
        result[mask] = grey_levels[i]

    # Make sure pixels == 255 (if any reach exactly 255) fall in the last class
    result[image == 255] = grey_levels[-1]

    return result