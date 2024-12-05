import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh từ file
img = cv2.imread('gggh.jpg')
print("Kích thước của ảnh gốc là:", img.shape)

# Lấy các kênh màu
R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]

# Chuyển đổi sang ảnh màu xám theo công thức
img_gray = 0.2989* R + 0.5870 * G + 0.1140  * B

def rgb_to_hsv(r, g, b):
    # Normalization of RGB values
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Calculate Cmax, Cmin, and Δ
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    # Hue calculation
    if delta == 0:
        h = 0
    elif cmax == r:
        h = 60 * ((g - b) / delta % 6)
    elif cmax == g:
        h = 60 * ((b - r) / delta + 2)
    elif cmax == b:
        h = 60 * ((r - g) / delta + 4)

    # Saturation calculation
    s = 0 if cmax == 0 else delta / cmax

    # Value calculation
    v = cmax

    return round(h % 360), round(s * 100), round(v * 100)

# Khởi tạo ảnh HSV với giá trị Hue, Saturation, và Value
img_hsv = np.zeros_like(img)

# Chuyển đổi từ không gian màu RGB sang HSV
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        r, g, b = img[i, j]
        h, s, v = rgb_to_hsv(r, g, b)
        img_hsv[i, j] = [h, s, v]

# Chuyển đổi không gian màu RGB sang YCbCr

Cr = 128 + 0.438*R - 0.336*G + 0.071*B
Cb = 128 - 0.148*R - 0.290*G - 0.438*B

# Hàm tăng cường độ tương phản
def contrast_stretching(image):
    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    normalized_image = image / 255.0

    # Đặt mức đen và mức trắng mong muốn (vd: 5% và 95% percentile)
    min_percentile = 5
    max_percentile = 95

    min_value = np.percentile(normalized_image, min_percentile)
    max_value = np.percentile(normalized_image, max_percentile)

    # Áp dụng công thức cắt mức
    contrast_stretched = np.clip((normalized_image - min_value) / (max_value - min_value), 0, 1)

    # Chuyển đổi về giá trị [0, 255]
    contrast_stretched = (contrast_stretched * 255).astype(np.uint8)

    return contrast_stretched

# Tính lược đồ mức xám
histogram = cv2.calcHist([img_gray.astype('uint8')], [0], None, [256], [0, 256])

# Hiển thị các ảnh
fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))

axs1[0, 0].imshow(img[:, :, ::-1])
axs1[0, 0].set_title('Ảnh gốc')

# Áp dụng contrast stretching cho ảnh mức xám
contrast_stretched_img_gray = contrast_stretching(img_gray)

# Hiển thị ảnh sau khi tăng cường độ tương phản
axs1[0, 1].imshow(img_gray, cmap='gray')
axs1[0, 1].set_title('Ảnh mức xám (Tăng cường độ tương phản)')

# Hiển thị tiêu đề cho ảnh HSV
axs1[1, 0].imshow(img_hsv)
axs1[1, 0].set_title('Ảnh HSV')

# Hiển thị tiêu đề cho ảnh YCBCR
axs1[1, 1].imshow(cv2.merge([img_gray, Cr, Cb]).astype(np.uint8))
axs1[1, 1].set_title('Ảnh YCBCR')

# Tạo một lưới subplot với 1 hàng và 3 cột
fig, axs = plt.subplots(1, 3, figsize=(10, 8))

# Hiển thị ảnh mức xám
axs[0].imshow(img_gray, cmap='gray')
axs[0].set_title('Ảnh Mức Xám')

# Hiển thị lược đồ mức xám
axs[1].plot(histogram, color='blue')
axs[1].set_title('Lược Đồ Mức Xám')
axs[1].set_xlabel('Giá Trị Pixel')
axs[1].set_ylabel('Số Lượng Pixel')

# Áp dụng contrast stretching cho ảnh mức xám
contrast_stretched_img_gray = contrast_stretching(img_gray)
axs[2].imshow(contrast_stretched_img_gray, cmap='gray')
axs[2].set_title('Tăng Cường Độ Tương Phản')

# Tăng khoảng cách giữa các subplot
plt.tight_layout()

plt.show()
