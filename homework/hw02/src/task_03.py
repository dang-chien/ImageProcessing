"""
Task 03 – Binary Template Matching (Find letter 'T')

Yêu cầu:
- Đọc ảnh actontBin.bin (256x256, ảnh nhị phân 0/255).
- Thiết kế template chữ 'T'.
- Tính match measure M2 tại mỗi pixel (nếu có đủ neighborhood).
- Sinh ảnh J1 (giá trị M2), sau đó ngưỡng để có J2 (mask nhị phân vị trí chữ T).
- Hiển thị và lưu kết quả.

Đầu vào:
- actontBin.bin

Đầu ra:
- J1 (ảnh grayscale biểu diễn độ match).
- J2 (ảnh binary các vị trí phát hiện 'T').
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.io_utils import read_bin_image, write_image


def create_template_T(size: int = 9) -> np.ndarray:
    """
    Tạo template chữ 'T' nhị phân.
    size: kích thước (nên lẻ, ví dụ 9x9).
    """
    template = np.zeros((size, size), dtype=np.uint8)
    # thanh ngang trên
    template[0, :] = 1
    # thanh dọc ở giữa
    template[:, size // 2] = 1
    return template


def match_measure_M2(patch: np.ndarray, template: np.ndarray) -> float:
    """
    Tính match measure M2 cho 1 patch và template.
    patch, template: ảnh nhị phân 0/1.
    """
    return np.sum(patch == template) / template.size


def template_matching(img: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Áp dụng Binary Template Matching cho toàn ảnh.
    """
    h, w = img.shape
    th, tw = template.shape
    pad_h, pad_w = th // 2, tw // 2

    J1 = np.zeros_like(img, dtype=np.float32)

    for i in range(pad_h, h - pad_h):
        for j in range(pad_w, w - pad_w):
            patch = img[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1]
            J1[i, j] = match_measure_M2(patch, template)

    return J1


if __name__ == "__main__":
    bin_path = os.path.join("hw02", "data", "bin", "actontBin.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Không tìm thấy file: {bin_path}")

    # Đọc ảnh nhị phân (0/255) và chuẩn hóa về 0/1
    img = read_bin_image(bin_path, width=256, height=256)
    img_bin = (img > 127).astype(np.uint8)

    # Tạo template chữ 'T'
    template = create_template_T(size=9)

    # Matching
    J1 = template_matching(img_bin, template)

    # Chuẩn hóa J1 về 0–255 để hiển thị
    J1_norm = (J1 * 255).astype(np.uint8)

    # Ngưỡng để tạo J2
    threshold = 0.8
    J2 = (J1 >= threshold).astype(np.uint8) * 255

    # Hiển thị kết quả
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Ảnh gốc")
    axs[0].axis("off")

    axs[1].imshow(J1_norm, cmap="gray")
    axs[1].set_title("Ảnh J1 (match measure M2)")
    axs[1].axis("off")

    axs[2].imshow(J2, cmap="gray")
    axs[2].set_title("Ảnh J2 (sau threshold)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

    # Lưu kết quả
    out_dir = os.path.join("hw02", "results")
    os.makedirs(out_dir, exist_ok=True)
    write_image(os.path.join(out_dir, "Task3_J1.png"), J1_norm)
    write_image(os.path.join(out_dir, "Task3_J2.png"), J2)
