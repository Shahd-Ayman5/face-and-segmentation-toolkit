import cv2
import numpy as np
import heapq


def agglomerative_segmentation(
    image,
    n_clusters=12,
    block_size=3,
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

    # every block is a cluster not pixel
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):

            y2, x2 = min(y+block_size, h), min(x+block_size, w)

            mask = np.zeros((h, w), dtype=bool)
            mask[y:y2, x:x2] = True

            pixels = img[y:y2, x:x2]
            mean_color = np.mean(pixels, axis=(0,1))

            # to not blend same colors if they are far apart, we add spatial info to the mean
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
        return np.sqrt(np.sum(dc**2) + alpha*np.sum(ds**2)) # how similar the colors are + how close they are (2 clusters)
    heap = []

    neighbors = {i: set() for i in clusters}

    # this loop finds neighboring clusters by looking at the label map and adds their distances to the heap
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

        #merge masks
        new_mask = c1["mask"] | c2["mask"]  # make c1 and c2 pixels part of the same cluster


        pixels = img[new_mask] # get all pixels in the new cluster
        new_color = np.mean(pixels, axis=0)  # compute new mean color of the new cluster

        ys, xs = np.where(new_mask) # get coordinates of all pixels in the new cluster
        cx = np.mean(xs)
        cy = np.mean(ys)

        clusters[i]["mask"] = new_mask
        clusters[i]["mean"] = np.array([*new_color, cx, cy])

        clusters[j]["active"] = False
        current -= 1  # removed one cluster, we have i and j merged into i

        # update neighbors, because new cluster i has new neighbors not like i and j
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
    output = np.zeros_like(img)   # make o/p image

    for c in clusters.values():
        if not c["active"]:
            continue

        color = np.clip(c["mean"][:3], 0, 255) # get mean color of cluster and clip to valid range
        output[c["mask"]] = color # set all pixels in cluster to mean color

    output = cv2.resize(output, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return output