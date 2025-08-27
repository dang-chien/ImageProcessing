import cv2
import numpy as np


# Load ảnh
img1 = cv2.imread("img1.jpg", cv2.IMREAD_GRAYSCALE)  # ảnh có mạch
img2 = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)  # ảnh nền

# Resize cùng kích thước
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Step 1: Subtraction (background subtraction)
sub = cv2.subtract(img1, img2)

# Step 2: Enhance contrast (Histogram Equalization)
enhanced = cv2.equalizeHist(sub)

# Ngoài ra có thể dùng CLAHE để tránh quá sáng tối
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_enhanced = clahe.apply(sub)

# Hiển thị
cv2.imshow("Original", img1)
cv2.imshow("Background", img2)
cv2.imshow("Subtracted", sub)
cv2.imshow("Enhanced Contrast (HistEq)", enhanced)
cv2.imshow("Enhanced Contrast (CLAHE)", clahe_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
