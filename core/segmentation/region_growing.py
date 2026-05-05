from __future__ import annotations

from collections import deque
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette – up to 12 visually distinct BGR colours for regions
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE_BGR = [
    (0,   0,   255),   # red
    (0,   255, 0  ),   # green
    (255, 0,   0  ),   # blue
    (0,   200, 255),   # yellow
    (255, 0,   200),   # magenta
    (255, 200, 0  ),   # cyan
    (50,  150, 255),   # orange
    (150, 0,   255),   # pink
    (0,   255, 150),   # lime
    (200, 50,  0  ),   # teal
    (100, 100, 255),   # salmon
    (255, 100, 100),   # sky-blue
]


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────

def region_growing(image: np.ndarray,
                   seeds: list[tuple[int, int]],
                   threshold: int = 15) -> np.ndarray:
    
    if len(seeds) == 0:
        return image.copy()

    is_colour  = image.ndim == 3
    has_alpha  = is_colour and image.shape[2] == 4

    h, w = image.shape[:2]

    # Working image: grayscale for distance comparisons
    if is_colour:
        gray = _to_gray(image)
    else:
        gray = image.copy()

    # Result: dim version of original so grown regions stand out
    if is_colour:
        result = (image.astype(np.float32) * 0.4).astype(np.uint8)
    else:
        # Convert gray to BGR so we can paint colour regions
        gray3   = np.stack([gray, gray, gray], axis=2)
        result  = (gray3.astype(np.float32) * 0.4).astype(np.uint8)
        if has_alpha:
            alpha   = np.full((h, w, 1), 255, dtype=np.uint8)
            result  = np.concatenate([result, alpha], axis=2)

    # Global visited mask – a pixel may only belong to one region
    visited = np.zeros((h, w), dtype=bool)

    for seed_idx, (sx, sy) in enumerate(seeds):
        sx, sy = int(sx), int(sy)

        # Validate seed position
        if not (0 <= sx < w and 0 <= sy < h):
            continue

        colour = _PALETTE_BGR[seed_idx % len(_PALETTE_BGR)]

        seed_intensity = int(gray[sy, sx])
        queue = deque()
        queue.append((sx, sy))

        while queue:
            cx, cy = queue.popleft()

            # Out-of-bounds or already visited
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            if visited[cy, cx]:
                continue

            # Intensity gate
            if abs(int(gray[cy, cx]) - seed_intensity) > threshold:
                continue

            # Accept this pixel
            visited[cy, cx] = True
            _paint_pixel(result, cy, cx, colour, has_alpha)

            # Push 4-connected neighbours
            queue.append((cx + 1, cy))
            queue.append((cx - 1, cy))
            queue.append((cx,     cy + 1))
            queue.append((cx,     cy - 1))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    """Luminance-weighted BGR→gray (handles both 3- and 4-channel)."""
    bgr = image[:, :, :3].astype(np.float32)
    # OpenCV default weights for BGR: 0.114 B, 0.587 G, 0.299 R
    gray = (0.114 * bgr[:, :, 0] +
            0.587 * bgr[:, :, 1] +
            0.299 * bgr[:, :, 2])
    return gray.astype(np.uint8)


def _paint_pixel(result: np.ndarray,
                 row: int, col: int,
                 colour: tuple[int, int, int],
                 has_alpha: bool) -> None:
    if has_alpha:
        result[row, col] = (*colour, 255)
    else:
        result[row, col] = colour