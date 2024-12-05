import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gggh.jpg')
print("kích thước của ảnh gốc là :", img.shape)

R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]
img_gray = 0.1140 * R + 0.5870 * G + 0.2989 * B  # Công thức ảnh màu xám

# Nhiễu Poisson
def add_poisson_noise(image):
    # Generate Poisson noise
    poisson_noise = np.random.poisson(image)
    # Clip values to ensure they stay within the valid range [0, 255]
    poisson_noise = np.clip(poisson_noise, 0, 255)
    return poisson_noise

# Add Poisson noise to the grayscale image
img_gray_noisy_poisson = add_poisson_noise(img_gray)

# Nhiễu Gauss
mean = 0
std_dev = 25  # You can adjust the standard deviation as needed
noise = np.random.normal(mean, std_dev, img_gray.shape)#(u, phương sai, ma trận)
img_gray_noisy = img_gray + noise
# Clip values to ensure they stay within the valid range [0, 255]
img_gray_noisy = np.clip(img_gray_noisy, 0, 255)

# Nhiễu muối tiêu
salt_pepper_prob = 0.02  # Probability of adding salt or pepper noise
salt_prob = 0.5  # Probability of adding salt noise (versus pepper)

# Create a mask for salt noise
salt_mask = np.random.rand(*img_gray.shape) < salt_pepper_prob * salt_prob
img_gray_with_salt = img_gray.copy()
img_gray_with_salt[salt_mask] = 255  # Set salt pixels to white

# Create a mask for pepper noise
pepper_mask = np.random.rand(*img_gray.shape) < salt_pepper_prob * (1 - salt_prob)
img_gray_with_pepper = img_gray_with_salt.copy()
img_gray_with_pepper[pepper_mask] = 0  # Set pepper pixels to black

# Tạo mặt nạ kích thước 3x3 cho lọc trung bình
kernel_size = (3, 3)
kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])

# Lọc nhiễu Gauss bằng phương pháp lọc trung bình
img_gray_filtered = cv2.filter2D(img_gray, -1, kernel)

#lọc nhiễu bằng phương pháp lọc trung vị
def median_filter(img, kernel_size):
    rows, cols = img.shape
    result = np.zeros_like(img)

    half_kernel = kernel_size // 2

    for i in range(half_kernel, rows - half_kernel):
        for j in range(half_kernel, cols - half_kernel):
            # Lấy các giá trị pixel trong kernel
            window = img[i - half_kernel:i + half_kernel + 1, j - half_kernel:j + half_kernel + 1]

            # Sắp xếp các giá trị pixel và lấy giá trị trung vị
            result[i, j] = np.median(window)

    return result

# Kích thước mặt nạ lọc trung vị (3x3)
kernel_size_median = 3

# Lọc trung vị ảnh màu xám chứa nhiễu muối tiêu
img_gray_filtered_median_custom = median_filter(img_gray_with_pepper, kernel_size_median)

plt.figure(figsize=(30,20))

plt.subplot(341)
plt.imshow(img[:,:,::-1])
plt.title('Đây là ảnh gốc')

plt.subplot(342)
plt.imshow(img_gray, cmap='gray')
plt.title('Ảnh màu xám')

plt.subplot(343)
plt.imshow(img_gray_noisy, cmap='gray')
plt.title('Ảnh màu xám với nhiễu Gauss')

plt.subplot(344)
plt.imshow(img_gray_with_pepper, cmap='gray')
plt.title('Ảnh màu xám với nhiễu salt & pepper')

plt.subplot(345)
plt.imshow(img_gray_noisy_poisson, cmap='gray')
plt.title('Ảnh màu xám với nhiễu Poisson')

plt.subplot(346)
plt.imshow(img_gray_filtered, cmap='gray')
plt.title('Ảnh sau khi lọc nhiễu Gauss bằng lọc trung bình')

# Hiển thị ảnh sau khi lọc trung vị
plt.subplot(347)
plt.imshow(img_gray_filtered_median_custom, cmap='gray')
plt.title('Ảnh sau khi lọc trung vị nhiễu Muối Tiêu')


plt.show()
