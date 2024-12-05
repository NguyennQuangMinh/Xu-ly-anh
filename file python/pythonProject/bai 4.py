import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh
img = cv2.imread('gggh.jpg')

# Hiển thị kích thước ảnh gốc
print("Kích thước của ảnh gốc là:", img.shape)

# Tách ảnh thành các kênh màu RGB
R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]

# Chuyển đổi ảnh sang ảnh xám bằng công thức được cung cấp
img_gray = 0.1140 * R + 0.5870 * G + 0.2989 * B

# Hiển thị Biến đổi Fourier của ảnh xám
F = np.fft.fft2(img_gray)
plt.imshow(np.log1p(np.abs(F)), cmap='gray')
plt.axis('OFF')
plt.title('Biến đổi Fourier của ảnh xám')
plt.show()

# Di chuyển thành phần tần số zero đến trung tâm
FShift = np.fft.fftshift(F)
plt.imshow(np.log1p(np.abs(FShift)), cmap='gray')
plt.axis('OFF')
plt.title('Spectrum đã dịch')
plt.show()

# Bộ lọc thông thấp lý tưởng
M, N = FShift.shape
H = np.zeros((M, N), dtype=np.float32)

D0 = 100
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
        if D <= D0:
            H[u, v] = 1
        else:
            H[u, v] = 0

# Hiển thị bộ lọc
plt.imshow(H, cmap='gray')
plt.axis('OFF')
plt.title('Bộ Lọc Thông Thấp Lý Tưởng')
plt.show()

# Áp dụng bộ lọc vào miền tần số
G = FShift * H

# Hiển thị spectrum đã lọc
plt.imshow(np.log1p(np.abs(G)), cmap='gray')
plt.axis('OFF')
plt.title('Spectrum đã lọc')
plt.show()

# Biến đổi Fourier nghịch đảo để nhận được ảnh đã lọc
img_filtered = np.fft.ifft2(np.fft.ifftshift(G)).real

# Hiển thị ảnh gốc và ảnh đã lọc cùng một lúc
plt.subplot(121)
plt.imshow(img[:,:,::-1])
plt.title('Ảnh Gốc')
plt.axis('OFF')

plt.subplot(122)
plt.imshow(img_filtered, cmap='gray')
plt.title('Ảnh Đã Lọc')
plt.axis('OFF')

plt.show()
