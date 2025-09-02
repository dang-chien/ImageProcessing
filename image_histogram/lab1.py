import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---- 1. Đọc ảnh grayscale bằng PIL ----
def read_image(path):
    img = Image.open(path).convert("L")  # chuyển sang grayscale
    return np.array(img)

# ---- 2. Tính histogram ----
def compute_histogram(img):
    hist = np.zeros(256, dtype=int)
    for pixel in img.flatten():
        hist[pixel] += 1
    return hist

# ---- 3. Cân bằng histogram (theo công thức CDF) ----
def histogram_equalization(img):
    # B1: tính histogram
    hist = compute_histogram(img)
    n_pixels = img.size

    # B2: tính xác suất (PDF)
    pdf = hist / n_pixels

    # B3: tính hàm tích lũy (CDF)
    cdf = np.cumsum(pdf)

    # B4: ánh xạ sang giá trị mới (scale về 0-255)
    transform_map = np.floor(255 * cdf).astype(np.uint8)

    # B5: tạo ảnh mới với giá trị mức xám đã biến đổi
    equalized_img = transform_map[img]

    return equalized_img, hist, compute_histogram(equalized_img)

# ---- 4. Test ----
if __name__ == "__main__":
    # đọc ảnh
    img = read_image("bay.jpg")

    # cân bằng histogram
    eq_img, hist_before, hist_after = histogram_equalization(img)

    # vẽ kết quả
    plt.figure(figsize=(12,6))

    plt.subplot(2,2,1)
    plt.imshow(img, cmap="gray")
    plt.title("Ảnh gốc")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.bar(range(256), hist_before, color="blue")
    plt.title("Histogram gốc")

    plt.subplot(2,2,3)
    plt.imshow(eq_img, cmap="gray")
    plt.title("Ảnh sau Histogram Equalization")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.bar(range(256), hist_after, color="green")
    plt.title("Histogram sau Equalization")

    plt.tight_layout()
    plt.show()
