import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('binary_img.png', cv2.IMREAD_GRAYSCALE)

# Define erosion function
def erosion(image, kernel):
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    eroded_img = np.zeros_like(image)

    for i in range(pad_h, padded_img.shape[0] - pad_h):
        for j in range(pad_w, padded_img.shape[1] - pad_w):
            region = padded_img[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            eroded_img[i - pad_h, j - pad_w] = np.min(region * kernel)

    return eroded_img

# Define dilation function
def dilation(image, kernel):
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    dilated_img = np.zeros_like(image)

    for i in range(pad_h, padded_img.shape[0] - pad_h):
        for j in range(pad_w, padded_img.shape[1] - pad_w):
            region = padded_img[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            dilated_img[i - pad_h, j - pad_w] = np.max(region * kernel)

    return dilated_img

# Define structuring elements
rect_kernel = np.ones((5, 5), dtype=np.uint8)

# Perform erosion and dilation
eroded_rect = erosion(img, rect_kernel)
dilated_rect = dilation(img, rect_kernel)

# Display results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(eroded_rect, cmap='gray')
plt.title('Eroded (Rectangular Kernel)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(dilated_rect, cmap='gray')
plt.title('Dilated (Rectangular Kernel)')
plt.axis('off')

plt.show()