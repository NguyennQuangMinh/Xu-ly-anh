import cv2
import numpy as np
from PIL import Image
import os

# Hàm tính toán Local Binary Patterns (LBP) của một pixel
def compute_lbp_pixel(img, x, y):
    center = img[x, y]
    code = 0
    if img[x-1, y-1] >= center:
        code |= 1
    if img[x-1, y] >= center:
        code |= 2
    if img[x-1, y+1] >= center:
        code |= 4
    if img[x, y+1] >= center:
        code |= 8
    if img[x+1, y+1] >= center:
        code |= 16
    if img[x+1, y] >= center:
        code |= 32
    if img[x+1, y-1] >= center:
        code |= 64
    if img[x, y-1] >= center:
        code |= 128
    return code

# Hàm tính toán histogram của một vùng ảnh sử dụng LBP
def compute_lbp_histogram(img, x, y, width, height):
    hist = np.zeros(256, dtype=np.int32)
    for i in range(x, x + width):
        for j in range(y, y + height):
            lbp_code = compute_lbp_pixel(img, i, j)
            hist[lbp_code] += 1
    return hist

# Hàm tính toán histogram LBP của toàn bộ ảnh
def compute_global_lbp_histogram(img):
    height, width = img.shape
    hist = np.zeros(256, dtype=np.int32)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            lbp_code = compute_lbp_pixel(img, i, j)
            hist[lbp_code] += 1
    return hist

# Hàm nhận dạng khuôn mặt sử dụng LBPH
def recognize_face(input_image, trained_histogram):
    input_histogram = compute_global_lbp_histogram(input_image)

    # Tính toán độ tương đồng giữa histogram của ảnh đầu vào và histogram đã huấn luyện
    similarity = cv2.compareHist(trained_histogram, input_histogram, cv2.HISTCMP_CHISQR)
    return similarity

# Path for face image database
path = 'dataset'
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Hàm để đọc dữ liệu huấn luyện và gán nhãn cho mỗi khuôn mặt
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)

# Tạo histogram đã huấn luyện từ dữ liệu huấn luyện
trained_histogram = np.zeros(256, dtype=np.int32)
for face in faces:
    trained_histogram += compute_global_lbp_histogram(face)

# Normalize histogram
trained_histogram = trained_histogram / np.sum(trained_histogram)

# Save trained histogram
np.save('trained_histogram.npy', trained_histogram)

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
