"""
Task 04 – Histogram Equalization

Yêu cầu:
- Đọc ảnh johnny.bin (256x256, 8-bit).
- Vẽ histogram cho ảnh gốc.
- Thực hiện histogram equalization.
- Vẽ histogram cho ảnh sau cân bằng.
- Hiển thị cả ảnh gốc và ảnh sau khi cân bằng cùng với histogram.
- Xuất ảnh kết quả ra file.

Đầu vào:
- johnny.bin (ảnh 256x256 grayscale, 8-bit).

Đầu ra:
- Histogram ảnh gốc.
- Histogram ảnh sau khi equalization.
- Ảnh kết quả sau xử lý.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.io_utils import read_bin_image, write_image

def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]

    img_eq = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    return img_eq.reshape(img.shape).astype(np.uint8)

if __name__ == "__main__":
    # Đường dẫn file ảnh
    bin_path = os.path.join("hw02", "data", "bin", "johnny.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Không tìm thấy file: {bin_path}")

    # Đọc ảnh gốc
    img = read_bin_image(bin_path, width=256, height=256)

    # Equalization
    img_eq = histogram_equalization(img)

    # Hiển thị tất cả trong 1 figure
    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.hist(img.ravel(), bins=256, range=(0,255))
    plt.title("Histogram - Original")

    plt.subplot(2,2,3)
    plt.imshow(img_eq, cmap="gray")
    plt.title("Equalized Image")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.hist(img_eq.ravel(), bins=256, range=(0,255))
    plt.title("Histogram - Equalized")

    plt.tight_layout()
    plt.show()

    # Lưu ảnh kết quả
    out_dir = os.path.join("hw02", "results")
    os.makedirs(out_dir, exist_ok=True)
    write_image(os.path.join(out_dir, "Task4_histogram_equalization.png"), img_eq)
