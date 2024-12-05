import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('gggh.jpg', 0)

def gaussian_kernel(size, sigma=1.4):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g

def convolve2D(image, kernel):
    kernel = np.rot90(kernel)
    kernel_size = kernel.shape[0]
    kernel_radius = kernel_size // 2
    result = np.zeros_like(image)


    for i in range(kernel_radius, image.shape[0] - kernel_radius):
        for j in range(kernel_radius, image.shape[1] - kernel_radius):
          vung_lan_can = image[i - kernel_radius : i + kernel_radius + 1, j - kernel_radius : j + kernel_radius + 1]

          result[i, j] = np.sum(vung_lan_can * kernel)

    return result

g = gaussian_kernel(5, 1.4)

filtered_image = convolve2D(img, g)

plt.imshow(filtered_image, cmap='gray')
plt.axis('off')
plt.show()
