import cv2
import numpy as np


def agglomerative_segmentation(
    image,
    n_clusters=8,
    block_size=4,
    progress_queue=None
):
    """
    Agglomerative segmentation (from scratch).

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color)

    n_clusters : int
        Desired number of clusters

    block_size : int
        Initial grouping size (to reduce complexity)

    Returns
    -------
    segmented : np.ndarray
    """

    img = image.copy()

    if img is None:
        raise ValueError("Input image is None")

    # Ensure 3 channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w, c = img.shape

    # 🔹 Step 1: initial clusters (blocks)
    clusters = []
    labels = -np.ones((h, w), dtype=int)

    cluster_id = 0

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img[y:y+block_size, x:x+block_size]

            mean_color = np.mean(block.reshape(-1, 3), axis=0)

            clusters.append({
                "id": cluster_id,
                "pixels": [(i, j)
                           for i in range(y, min(y+block_size, h))
                           for j in range(x, min(x+block_size, w))],
                "mean": mean_color
            })

            for i in range(y, min(y+block_size, h)):
                for j in range(x, min(x+block_size, w)):
                    labels[i, j] = cluster_id

            cluster_id += 1

    total_merges = len(clusters) - n_clusters
    merge_count = 0

    print(f"[Agglomerative] Initial clusters: {len(clusters)}")

    # 🔹 Step 2: merging
    while len(clusters) > n_clusters:

        min_dist = float("inf")
        pair = (0, 1)

        # find closest pair
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = np.linalg.norm(
                    clusters[i]["mean"] - clusters[j]["mean"]
                )
                if dist < min_dist:
                    min_dist = dist
                    pair = (i, j)

        i, j = pair
        c1, c2 = clusters[i], clusters[j]

        # merge
        new_pixels = c1["pixels"] + c2["pixels"]
        new_mean = np.mean(
            [img[p] for p in new_pixels], axis=0
        )

        new_cluster = {
            "id": c1["id"],
            "pixels": new_pixels,
            "mean": new_mean
        }

        # update labels
        for (y, x) in c2["pixels"]:
            labels[y, x] = c1["id"]

        # replace clusters
        clusters[i] = new_cluster
        clusters.pop(j)

        merge_count += 1

        # progress
        if progress_queue is not None and total_merges > 0:
            progress = int((merge_count / total_merges) * 100)
            progress_queue.put(("progress", progress))

        if merge_count % 10 == 0:
            print(f"[Agglomerative] Merges: {merge_count}")

    # 🔹 Step 3: reconstruct image
    output = np.zeros_like(img)

    for cluster in clusters:
        color = cluster["mean"]
        for (y, x) in cluster["pixels"]:
            output[y, x] = color

    output = np.clip(output, 0, 255).astype(np.uint8)

    print("[Agglomerative] Done")

    if progress_queue is not None:
        progress_queue.put(("progress", 100))

    return output


