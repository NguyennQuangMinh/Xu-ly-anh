import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh và chuyển sang ảnh mức xám
img = cv2.imread('gggh.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

# Tạo ma trận mask Gaussian
kernel = gaussian_kernel(5, sigma=1.4)
kernel = (kernel * 255).astype(np.uint8)  # Chuyển đổi ma trận thành kiểu dữ liệu uint8
print(kernel)

# Tính kích thước của ảnh mức xám
height, width = img_gray.shape

# Tính kích thước của mask
k_size = kernel.shape[0]
k_radius = k_size // 2

# Tạo ma trận mới cho ảnh kết quả
img_blurred = np.zeros_like(img_gray)

# Áp dụng phép tích chập
for y in range(k_radius, height - k_radius):
    for x in range(k_radius, width - k_radius):
        roi = img_gray[y - k_radius:y + k_radius + 1, x - k_radius:x + k_radius + 1]
        img_blurred[y, x] = np.sum(roi * kernel)

# Hiển thị ảnh mức xám sau khi thực hiện phép tích chập
plt.subplot(121)
plt.imshow(img_gray, cmap='gray')
plt.title('Ảnh mức xám gốc')

plt.subplot(122)
plt.imshow(img_blurred, cmap='gray')
plt.title('Ảnh mức xám sau khi làm trơn')

plt.show()
