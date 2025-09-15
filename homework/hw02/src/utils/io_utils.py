# src/utils/io_utils.py
# ==========================================================
# IO utilities for Image Processing labs
# - Đọc/ghi ảnh .bin (grayscale 8-bit)
# - Đọc/ghi ảnh JPG/PNG bằng OpenCV
# - Chuyển kênh & đảm bảo dtype phục vụ Task 1→3
# ==========================================================
from __future__ import annotations
import os
import numpy as np
import cv2


def read_bin_image(path: str, width: int = 256, height: int = 256,
                   dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Đọc ảnh nhị phân RAW (.bin) dạng grayscale 8-bit.

    Parameters
    ----------
    path : str
        Đường dẫn file .bin (không có header).
    width, height : int
        Kích thước ảnh cần reshape (mặc định 256x256 cho bài HW01).
    dtype : np.dtype
        Kiểu dữ liệu mỗi pixel (mặc định uint8).

    Returns
    -------
    np.ndarray
        Ảnh 2D (height, width), dtype=uint8.

    Raises
    ------
    FileNotFoundError: nếu path không tồn tại.
    ValueError: nếu số byte trong file không khớp width*height.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    data = np.fromfile(path, dtype=dtype)
    expected = width * height
    if data.size != expected:
        raise ValueError(
            f"Raw size mismatch: got {data.size} bytes, expected {expected} "
            f"for image {width}x{height}."
        )
    return data.reshape((height, width))


def write_bin_image(img: np.ndarray, path: str) -> None:
    """
    Ghi ảnh grayscale 2D ra file .bin (RAW).

    Notes
    -----
    - Không ghi header; chỉ ghi byte pixel.
    - Ảnh phải là 2D; sẽ được ép về uint8 trước khi ghi.
    """
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError("write_bin_image chỉ nhận ảnh 2D (grayscale).")
    arr = ensure_uint8(arr)
    arr.tofile(path)


def read_image(path: str, as_gray: bool = False) -> np.ndarray:
    """
    Đọc ảnh JPG/PNG bằng OpenCV.

    Parameters
    ----------
    as_gray : bool
        True → trả ảnh 2D grayscale.
        False → trả ảnh màu BGR (chuẩn OpenCV).

    Returns
    -------
    np.ndarray
        2D (gray) hoặc 3D (BGR).
    """
    if as_gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def write_image(path: str, img: np.ndarray) -> None:
    """
    Ghi ảnh (JPG/PNG, v.v.) bằng OpenCV.

    Raises
    ------
    IOError: nếu ghi file thất bại (ví dụ: thư mục không tồn tại).
    """
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """
    Đảm bảo ảnh ở dạng grayscale 2D.
    - Nếu 2D → trả về nguyên trạng.
    - Nếu 3D (BGR) → convert sang GRAY bằng cv2.cvtColor.
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError("Unsupported image ndim for ensure_gray.")


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """
    Ép ảnh về dtype=uint8 an toàn:
    - Clip vào [0,255] rồi astype(np.uint8)
    """
    return np.clip(img, 0, 255).astype(np.uint8)


def split_bgr(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tách kênh B, G, R từ ảnh màu BGR (OpenCV).
    Hữu ích cho Task 3 khi thao tác band riêng lẻ.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("split_bgr yêu cầu ảnh màu BGR 3 kênh.")
    b, g, r = cv2.split(img[:, :, :3])
    return b, g, r


def merge_bgr(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Gộp 3 kênh B, G, R thành ảnh màu BGR.
    """
    return cv2.merge((b, g, r))


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Chuyển BGR → RGB (để vẽ đúng màu bằng matplotlib)."""
    if img.ndim == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    return img
