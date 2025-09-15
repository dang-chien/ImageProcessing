# ==========================================================
# Task: Mammogram image processing
# (a) Thresholding Mammogram.bin -> Binary image
# (b) Approximate Contour Image Generation
# (c) Discussion: Chain code for contour representation
# ==========================================================

# hw02/src/task_01.py
# Xử lý ảnh Mammogram: threshold -> contour -> lưu & hiển thị
import os
import numpy as np
import cv2

from utils.io_utils import read_bin_image, write_image
from utils.display_utils import show_images_grid

# -------------------------
# (a) Ngưỡng (hỗ trợ Otsu nếu threshold=None)
# -------------------------
def threshold_image(img: np.ndarray, threshold: int | None = None) -> tuple[np.ndarray, float]:
    """
    Nếu threshold is None -> dùng Otsu để tự động chọn ngưỡng.
    Trả về (binary_img, used_threshold)
    """
    if threshold is None:
        # Otsu: trả về ret = ngưỡng được chọn, và ảnh nhị phân
        ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bin_img.astype(np.uint8), float(ret)
    else:
        bin_img = np.where(img > threshold, 255, 0).astype(np.uint8)
        return bin_img, float(threshold)

# -------------------------
# (b) Sinh ảnh contour xấp xỉ
# -------------------------
def contour_image(binary_img: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """
    connectivity: 4 hoặc 8
    Trả về ảnh contour (255 = biên, 0 = không)
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity phải là 4 hoặc 8")
    h, w = binary_img.shape
    contour = np.zeros_like(binary_img)

    if connectivity == 4:
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if binary_img[i, j] == 255:
                    if (binary_img[i-1, j] == 0 or
                        binary_img[i+1, j] == 0 or
                        binary_img[i, j-1] == 0 or
                        binary_img[i, j+1] == 0):
                        contour[i, j] = 255
    else:  # 8-neighbors
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if binary_img[i, j] == 255:
                    neighbors = binary_img[i-1:i+2, j-1:j+2]
                    if np.any(neighbors == 0):
                        contour[i, j] = 255
    return contour

# -------------------------
# (c) Bàn luận ngắn về chain code
# -------------------------
def chain_code_discussion() -> str:
    return (
        "Chain code có thể dùng để biểu diễn đường biên chính nếu ta có một "
        "chuỗi biên liên thông (single connected boundary). Nếu ảnh contour có "
        "nhiều mảnh rời rạc hoặc biên dày, cần tiền xử lý (lọc mảnh nhỏ, làm mượt, "
        "chọn contour lớn nhất) trước khi dùng chain code."
    )

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Chú ý: chạy từ thư mục gốc project (homework/)
    # cd <...>/homework
    # python -m hw02.src.task_01

    # Đường dẫn tương đối (so với cwd = homework/)
    bin_path = os.path.join("hw02", "data", "bin", "mammogram.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Không tìm thấy file: {bin_path}")

    # 1) Đọc ảnh RAW (256x256)
    img = read_bin_image(bin_path, width=256, height=256, dtype=np.uint32).astype(np.uint8)

    # 2) Ngưỡng: None => dùng Otsu tự động. Nếu muốn cố định, truyền threshold=int
    binary, used_T = threshold_image(img, threshold=None)

    # 3) Sinh contour (dùng 4-connectivity theo đề, đổi thành 8 nếu muốn)
    contour = contour_image(binary, connectivity=4)

    # 4) Lưu kết quả (tạo thư mục results nếu cần)
    out_dir = os.path.join("hw02", "results")
    os.makedirs(out_dir, exist_ok=True)
    write_image(os.path.join(out_dir, "Task1_mammogram_binary.png"), binary)
    write_image(os.path.join(out_dir, "Task1_mammogram_contour.png"), contour)

    # 5) Hiển thị
    show_images_grid(
        [img, binary, contour],
        titles=["Ảnh gốc", f"Ảnh nhị phân (T={used_T:.2f})", "Ảnh contour"],
        cols=3
    )

    # 6) In bàn luận
    print("Bàn luận về chain code:")
    print(chain_code_discussion())


