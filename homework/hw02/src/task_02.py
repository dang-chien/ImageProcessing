"""
Task 02 – Contrast Stretching & Histogram

Yêu cầu:
- Đọc ảnh lady.bin (256x256, 8-bit).
- Vẽ biểu đồ histogram cho ảnh gốc.
- Thực hiện full-scale contrast stretching (kéo dãn độ tương phản từ [min, max] về [0, 255]).
- Vẽ histogram cho ảnh sau khi kéo dãn.
- Xuất ảnh kết quả ra file.

Đầu vào:
- lady.bin (ảnh 256x256 grayscale, 8-bit).

Đầu ra:
- Histogram ảnh gốc.
- Histogram ảnh sau khi contrast stretching.
- Ảnh kết quả sau xử lý.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.io_utils import read_bin_image, write_image


def contrast_stretch(img):
    min_val, max_val = np.min(img), np.max(img)
    stretched = (img - min_val) * 255.0 / (max_val - min_val)
    return stretched.astype(np.uint8)


if __name__ == "__main__":
    # Đường dẫn file lady.bin
    bin_path = os.path.join("hw02", "data", "bin", "johnny.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Không tìm thấy file: {bin_path}")

    img = read_bin_image(bin_path, width=256, height=256)
    stretched_img = contrast_stretch(img)

    # Hiển thị tất cả trong 1 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Ảnh gốc
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Ảnh gốc")
    axes[0, 0].axis("off")

    # Histogram gốc
    axes[0, 1].hist(img.ravel(), bins=256, range=(0, 255), color="gray")
    axes[0, 1].set_title("Histogram ảnh gốc")

    # Ảnh sau kéo dãn
    axes[1, 0].imshow(stretched_img, cmap="gray")
    axes[1, 0].set_title("Ảnh sau Contrast Stretching")
    axes[1, 0].axis("off")

    # Histogram sau kéo dãn
    axes[1, 1].hist(stretched_img.ravel(), bins=256, range=(0, 255), color="gray")
    axes[1, 1].set_title("Histogram sau Contrast Stretching")

    plt.tight_layout()
    plt.show()

    # Xuất ảnh kết quả
    out_dir = os.path.join("hw02", "results")
    os.makedirs(out_dir, exist_ok=True)
    write_image(os.path.join(out_dir, "Task2_contrast_stretching.png"), stretched_img)
