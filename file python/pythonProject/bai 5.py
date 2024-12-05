import cv2
import numpy as np
from matplotlib import pyplot as plt

input_img = cv2.imread('binary_img.png', cv2.IMREAD_GRAYSCALE)
def erosion(img):
    se = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=np.uint8)  # Ensure dtype is uint8 for consistency
    result = np.zeros_like(img)  # Initialize result image
    height, width = img.shape
    img_pad = np.pad(img, ((1, 1), (1, 1)), mode='constant')  # Padding the input image
    for i in range(1, height+1):
        for j in range(1, width+1):
            roi = img_pad[i-1:i+2, j-1:j+2]  # Extract the region of interest
            # Perform erosion by AND operation
            if np.all(roi[se == 1]):  # Check if all non-zero elements in the ROI are 1
                result[i-1, j-1] = 255  # Set result pixel to 255 (white)
    return result

ret,binary_img = cv2.threshold(input_img,127,255,cv2.THRESH_BINARY)
print(binary_img.shape)
# print(binary_img[1:10,1:10])
result=binary_img.copy()
erode_img = erosion(binary_img)
plt.subplot(211)
plt.imshow(binary_img,cmap="gray")
plt.title("day la anh goc")
plt.subplot(212)
plt.imshow(erode_img,cmap="gray")
plt.title("anh soi mon")
plt.tight_layout()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
