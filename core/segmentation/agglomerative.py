# import cv2
# import numpy as np


# def agglomerative_segmentation(
#     image,
#     n_clusters=8,
#     block_size=4,
#     progress_queue=None
# ):
#     """
#     Agglomerative segmentation (from scratch).

#     Parameters
#     ----------
#     image : np.ndarray
#         Input image (grayscale or color)

#     n_clusters : int
#         Desired number of clusters

#     block_size : int
#         Initial grouping size (to reduce complexity)

#     Returns
#     -------
#     segmented : np.ndarray
#     """

#     img = image.copy()

#     if img is None:
#         raise ValueError("Input image is None")

#     # Ensure 3 channels
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     elif img.shape[2] == 4:
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

#     h, w, c = img.shape

#     # 🔹 Step 1: initial clusters (blocks)
#     clusters = []
#     labels = -np.ones((h, w), dtype=int)

#     cluster_id = 0

#     for y in range(0, h, block_size):
#         for x in range(0, w, block_size):
#             block = img[y:y+block_size, x:x+block_size]

#             mean_color = np.mean(block.reshape(-1, 3), axis=0)

#             clusters.append({
#                 "id": cluster_id,
#                 "pixels": [(i, j)
#                            for i in range(y, min(y+block_size, h))
#                            for j in range(x, min(x+block_size, w))],
#                 "mean": mean_color
#             })

#             for i in range(y, min(y+block_size, h)):
#                 for j in range(x, min(x+block_size, w)):
#                     labels[i, j] = cluster_id

#             cluster_id += 1

#     total_merges = len(clusters) - n_clusters
#     merge_count = 0

#     print(f"[Agglomerative] Initial clusters: {len(clusters)}")

#     # 🔹 Step 2: merging
#     while len(clusters) > n_clusters:

#         min_dist = float("inf")
#         pair = (0, 1)

#         # find closest pair
#         for i in range(len(clusters)):
#             for j in range(i + 1, len(clusters)):
#                 dist = np.linalg.norm(
#                     clusters[i]["mean"] - clusters[j]["mean"]
#                 )
#                 if dist < min_dist:
#                     min_dist = dist
#                     pair = (i, j)

#         i, j = pair
#         c1, c2 = clusters[i], clusters[j]

#         # merge
#         new_pixels = c1["pixels"] + c2["pixels"]
#         new_mean = np.mean(
#             [img[p] for p in new_pixels], axis=0
#         )

#         new_cluster = {
#             "id": c1["id"],
#             "pixels": new_pixels,
#             "mean": new_mean
#         }

#         # update labels
#         for (y, x) in c2["pixels"]:
#             labels[y, x] = c1["id"]

#         # replace clusters
#         clusters[i] = new_cluster
#         clusters.pop(j)

#         merge_count += 1

#         # progress
#         if progress_queue is not None and total_merges > 0:
#             progress = int((merge_count / total_merges) * 100)
#             progress_queue.put(("progress", progress))

#         if merge_count % 10 == 0:
#             print(f"[Agglomerative] Merges: {merge_count}")

#     # 🔹 Step 3: reconstruct image
#     output = np.zeros_like(img)

#     for cluster in clusters:
#         color = cluster["mean"]
#         for (y, x) in cluster["pixels"]:
#             output[y, x] = color

#     output = np.clip(output, 0, 255).astype(np.uint8)

#     print("[Agglomerative] Done")

#     if progress_queue is not None:
#         progress_queue.put(("progress", 100))

#     return output

import cv2
import numpy as np
import heapq


def agglomerative_segmentation(
    image,
    n_clusters=12,
    block_size=4,
    alpha=0.2,
    downscale=300,
    progress_queue=None
):

    img = image.copy()

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    orig_h, orig_w = img.shape[:2]

    img = cv2.resize(img, (downscale, downscale))
    h, w = img.shape[:2]

    # ---------------------------
    # 1. init clusters (mask-based)
    # ---------------------------
    clusters = {}
    label_map = -np.ones((h, w), dtype=np.int32)

    cid = 0

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):

            y2, x2 = min(y+block_size, h), min(x+block_size, w)

            mask = np.zeros((h, w), dtype=bool)
            mask[y:y2, x:x2] = True

            pixels = img[y:y2, x:x2]
            mean_color = np.mean(pixels, axis=(0,1))

            cx = (x + x2) / 2
            cy = (y + y2) / 2

            clusters[cid] = {
                "mask": mask,
                "mean": np.array([*mean_color, cx, cy]),
                "active": True
            }

            label_map[y:y2, x:x2] = cid
            cid += 1

    # ---------------------------
    # 2. adjacency + heap
    # ---------------------------
    def distance(c1, c2):
        dc = c1[:3] - c2[:3]
        ds = c1[3:] - c2[3:]
        return np.sqrt(np.sum(dc**2) + alpha*np.sum(ds**2))

    neighbors = {i: set() for i in clusters}
    heap = []

    for y in range(h):
        for x in range(w):
            c = label_map[y, x]

            if x+1 < w:
                c2 = label_map[y, x+1]
                if c != c2:
                    neighbors[c].add(c2)
                    neighbors[c2].add(c)

            if y+1 < h:
                c2 = label_map[y+1, x]
                if c != c2:
                    neighbors[c].add(c2)
                    neighbors[c2].add(c)

    for i in neighbors:
        for j in neighbors[i]:
            if i < j:
                d = distance(clusters[i]["mean"], clusters[j]["mean"])
                heapq.heappush(heap, (d, i, j))

    # ---------------------------
    # 3. merge loop
    # ---------------------------
    total = len(clusters)
    current = total

    while current > n_clusters and heap:

        d, i, j = heapq.heappop(heap)

        if not clusters[i]["active"] or not clusters[j]["active"]:
            continue

        c1 = clusters[i]
        c2 = clusters[j]

        #merge masks (THE FIX)
        new_mask = c1["mask"] | c2["mask"]

        pixels = img[new_mask]
        new_color = np.mean(pixels, axis=0)

        ys, xs = np.where(new_mask)
        cx = np.mean(xs)
        cy = np.mean(ys)

        clusters[i]["mask"] = new_mask
        clusters[i]["mean"] = np.array([*new_color, cx, cy])

        clusters[j]["active"] = False
        current -= 1

        # update neighbors
        for n in neighbors[j]:
            if clusters[n]["active"] and n != i:
                neighbors[i].add(n)
                neighbors[n].add(i)

                d_new = distance(clusters[i]["mean"], clusters[n]["mean"])
                heapq.heappush(heap, (d_new, i, n))

        neighbors[j].clear()

        # progress
        if progress_queue:
            prog = int((1 - current / total) * 100)
            progress_queue.put(("progress", prog))

    # ---------------------------
    # 4. reconstruct
    # ---------------------------
    output = np.zeros_like(img)

    for c in clusters.values():
        if not c["active"]:
            continue

        color = np.clip(c["mean"][:3], 0, 255)
        output[c["mask"]] = color

    output = cv2.resize(output, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return output