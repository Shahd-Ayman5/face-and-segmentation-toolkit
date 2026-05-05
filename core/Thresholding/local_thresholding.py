import numpy as np


def local_threshold(image, block_size=11, C=2):
   
    # Validate block_size(odd)
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")

    pad = block_size // 2

    # padding the image to handle borders
    padded = np.pad(image, pad, mode='reflect')

    binary_image = np.zeros_like(image)

    rows, cols = image.shape

    # loop through each pixel in the image
    for i in range(rows):
        for j in range(cols):

            # get the local window around the pixel
            window = padded[i:i + block_size, j:j + block_size]

            # mean 
            local_mean = np.mean(window)

            # threshold
            T = local_mean - C

            # decide if the pixel is black or white
            if image[i, j] > T:  # if pixel value is greater than threshold, set to white
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0

    return binary_image


