import cv2
import numpy as np


def otsu_threshold(image):
    
    # Compute histogram (256 bins for grayscale)
    hist = np.zeros(256)

    for pixel in image.flatten():
        hist[pixel] += 1

    total_pixels = image.shape[0] * image.shape[1]

    # Normalize histogram to probabilities
    prob = hist / total_pixels

    best_threshold = 0
    max_between_variance = 0

    w0 = 0          # weight background
    mu0 = 0         # mean background
    total_mean = 0

    # Compute total mean intensity
    for i in range(256):
        total_mean += i * prob[i]    # mean intensity

    cumulative_mean = 0

    for t in range(256):
        w0 += prob[t]               # background probability
        w1 = 1 - w0                 # foreground probability

        if w0 == 0 or w1 == 0:
            continue

        cumulative_mean += t * prob[t]
        mu0 = cumulative_mean / w0
        mu1 = (total_mean - cumulative_mean) / w1

        # Between-class variance
        between_variance = w0 * w1 * ((mu0 - mu1) ** 2)

        if between_variance > max_between_variance:
            max_between_variance = between_variance
            best_threshold = t

    # Apply threshold
    binary_image = np.zeros_like(image)

    binary_image[image > best_threshold] = 255
    binary_image[image <= best_threshold] = 0

    return best_threshold, binary_image
