import cv2
import numpy as np
from matplotlib import pyplot as plt

def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def sobel_operator(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    return magnitude, gradient_direction

def non_max_suppression(magnitude, gradient_direction):
    height, width = magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.uint8)
    angle_quantized = (np.round(gradient_direction * (5.0 / np.pi)) % 5).astype(int)
    for i in range(1, height-1):
        for j in range(1, width-1):
            if angle_quantized[i, j] == 0:
                if magnitude[i, j] > magnitude[i, j-1] and magnitude[i, j] > magnitude[i, j+1]:
                    suppressed[i, j] = magnitude[i, j]
            elif angle_quantized[i, j] == 1:
                if magnitude[i, j] > magnitude[i-1, j+1] and magnitude[i, j] > magnitude[i+1, j-1]:
                    suppressed[i, j] = magnitude[i, j]
            elif angle_quantized[i, j] == 2:
                if magnitude[i, j] > magnitude[i-1, j] and magnitude[i, j] > magnitude[i+1, j]:
                    suppressed[i, j] = magnitude[i, j]
            elif angle_quantized[i, j] == 3:
                if magnitude[i, j] > magnitude[i-1, j-1] and magnitude[i, j] > magnitude[i+1, j+1]:
                    suppressed[i, j] = magnitude[i, j]
    return suppressed

def threshold(image, low_threshold, high_threshold):
    strong_edges = (image > high_threshold)
    weak_edges = (image >= low_threshold) & (image <= high_threshold)
    return strong_edges, weak_edges

def hysteresis(strong_edges, weak_edges):
    height, width = strong_edges.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            if weak_edges[i, j]:
                if strong_edges[i-1:i+2, j-1:j+2].any():
                    strong_edges[i, j] = True
                else:
                    weak_edges[i, j] = False
    return strong_edges

def canny_edge_detection(image, kernel_size, low_threshold, high_threshold):
    blurred_image = gaussian_blur(image, kernel_size)
    magnitude, gradient_direction = sobel_operator(blurred_image)
    suppressed = non_max_suppression(magnitude, gradient_direction)
    strong_edges, weak_edges = threshold(suppressed, low_threshold, high_threshold)
    edges = hysteresis(strong_edges, weak_edges)
    return edges

# Thay đổi đường dẫn và các ngưỡng tùy thích
image_path = 'xyz.jpg'
kernel_size = 5
low_threshold = 100
high_threshold = 60

# Đọc ảnh
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Áp dụng phương pháp Canny
edges = canny_edge_detection(image, kernel_size, low_threshold, high_threshold)

# Hiển thị ảnh mức xám sau khi thực hiện phép tích chập
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc')

plt.subplot(122)
plt.imshow(np.uint8(edges) * 255, cmap='gray')
plt.title('Canny Edges')

plt.show()