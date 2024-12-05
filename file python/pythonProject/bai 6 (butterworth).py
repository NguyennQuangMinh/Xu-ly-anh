import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
image = cv2.imread('gggh.jpg', cv2.IMREAD_GRAYSCALE)

# Thực hiện biến đổi Fourier
f_transform = np.fft.fft2(image)
fshift = np.fft.fftshift(f_transform)

# Tính toán kích thước ảnh
rows, cols = image.shape
crow, ccol = rows / 2, cols / 2

# Tính toán khoảng cách đến trung tâm của không gian tần số
u = np.arange(-crow, crow)
v = np.arange(-ccol, ccol)
U, V = np.meshgrid(u, v)
D = np.sqrt(U**2 + V**2)

# Thiết lập các tham số của bộ lọc Butterworth
D0 = 5  # Tần số cắt
n = 2   # Bậc của bộ lọc Butterworth

# Resize D để khớp với kích thước của fshift
D_resized = cv2.resize(D, (cols, rows))

# Áp dụng công thức bộ lọc Butterworth
H = 1 / (1 + (D_resized / D0)**(2*n))

# Áp dụng bộ lọc vào miền tần số
filtered_fshift = fshift * H

# Biến đổi Fourier ngược
f_ishift = np.fft.ifftshift(filtered_fshift)
image_filtered = np.fft.ifft2(f_ishift)
image_filtered = np.abs(image_filtered)

# Biến đổi Fourier dịch chuyển (DFT shift)
dft_shift_image = np.fft.fftshift(np.log(1 + np.abs(fshift)))

# Biến đổi Fourier nghịch đảo dịch chuyển (IDFT shift)
idft_shift_image = np.abs(np.fft.ifft2(filtered_fshift))

# Hiển thị ảnh gốc và các biến đổi
plt.figure(figsize=(15, 10))

plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Ảnh gốc')
plt.subplot(232), plt.imshow(dft_shift_image, cmap='gray'), plt.title('DFT shift')
plt.subplot(233), plt.imshow(idft_shift_image, cmap='gray'), plt.title('IDFT shift')


plt.subplot(234), plt.imshow(image_filtered, cmap='gray'), plt.title('Lọc thông thấp Butterworth')

plt.show()
