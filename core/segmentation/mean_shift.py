import cv2
import numpy as np


def mean_shift_segmentation(
    image,
    spatial_radius=10,
    color_radius=30,
    max_iter=10,
    progress_queue=None
):
    """
    Mean Shift segmentation.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color)

    spatial_radius : int
        Radius for spatial neighborhood

    color_radius : int
        Radius for color similarity

    max_iter : int
        Max iterations for shifting

    Returns
    -------
    segmented : np.ndarray
        Segmented image
    """

    img = image.copy()

    if img is None:
        raise ValueError("Input image is None")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        # Support PNG/BGRA inputs by dropping alpha.
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Unsupported image shape for mean shift: {img.shape}")

    h, w, c = img.shape
    segmented = img.astype(np.float32)

    # Create feature space: [x, y, r, g, b]
    features = []

    for y in range(h):
        for x in range(w):
            r, g, b = segmented[y, x]
            features.append([x, y, r, g, b])

    features = np.array(features, dtype=np.float32)

    total_points = len(features)
    if total_points > 0:
        print(f"[Mean Shift] Processing started: {total_points} points")
        if progress_queue is not None:
            progress_queue.put(("progress", 0))

    # Print progress every 5% (and at the end).
    next_progress = 5

    for i in range(len(features)):
        point = features[i].copy()

        for _ in range(max_iter):
            # Compute distances
            spatial_dist = np.linalg.norm(
                features[:, :2] - point[:2], axis=1
            )

            color_dist = np.linalg.norm(
                features[:, 2:] - point[2:], axis=1
            )

            # Select neighbors
            neighbors = features[
                (spatial_dist < spatial_radius) &
                (color_dist < color_radius)
            ]

            if len(neighbors) == 0:
                break

            new_point = np.mean(neighbors, axis=0)

            shift = np.linalg.norm(new_point - point)
            point = new_point

            if shift < 1:
                break

        features[i] = point

        if total_points > 0:
            progress = int(((i + 1) / total_points) * 100)
            if progress >= next_progress or i == total_points - 1:
                print(f"[Mean Shift] Progress: {progress}%")
                if progress_queue is not None:
                    progress_queue.put(("progress", progress))
                while next_progress <= progress:
                    next_progress += 5

    # Reconstruct image
    output = np.zeros_like(segmented)

    idx = 0
    for y in range(h):
        for x in range(w):
            output[y, x] = features[idx, 2:]
            idx += 1

    output = np.clip(output, 0, 255).astype(np.uint8)

    print("[Mean Shift] Processing complete")
    if progress_queue is not None:
        progress_queue.put(("progress", 100))

    return output


if __name__ == "__main__":
    img = cv2.imread("image.jpg")

    result = mean_shift_segmentation(
        img,
        spatial_radius=12,
        color_radius=25,
        max_iter=10
    )

    cv2.imshow("Original", img)
    cv2.imshow("Mean Shift", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()