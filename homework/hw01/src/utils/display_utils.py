# src/utils/display_utils.py
# ==========================================================
# Display utilities cho việc visualize ảnh
# - Hiển thị 1 ảnh hoặc nhiều ảnh (grid) bằng matplotlib
# - Tùy chọn chuyển BGR→RGB tự động khi vẽ
# - Tùy chọn hiển thị bằng cửa sổ OpenCV nếu cần
# ==========================================================
from __future__ import annotations
import math
from typing import Iterable, Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2

from .io_utils import bgr_to_rgb


def show_image(img: np.ndarray, title: str = "Image",
               assume_bgr: bool = True, axis_off: bool = True) -> None:
    """
    Hiển thị 1 ảnh bằng matplotlib.

    Parameters
    ----------
    img : np.ndarray
        2D (grayscale) hoặc 3D (BGR/RGB).
    title : str
        Tiêu đề figure.
    assume_bgr : bool
        True nếu ảnh màu đến từ cv2.imread (BGR) → sẽ auto chuyển RGB.
        False nếu ảnh đã ở RGB.
    axis_off : bool
        Ẩn trục cho gọn.
    """
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(bgr_to_rgb(img) if assume_bgr else img)
    plt.title(title)
    if axis_off:
        plt.axis("off")
    plt.show()


def show_images_grid(
    images: Sequence[np.ndarray],
    titles: Optional[Sequence[str]] = None,
    cols: int = 2,
    figsize: Tuple[int, int] = (12, 6),
    assume_bgr: bool = True,
) -> None:
    """
    Hiển thị nhiều ảnh theo dạng lưới (grid).

    Parameters
    ----------
    images : list[np.ndarray]
        Danh sách ảnh (2D hoặc 3D).
    titles : list[str] | None
        Tiêu đề tương ứng từng ảnh (tuỳ chọn).
    cols : int
        Số cột của grid.
    figsize : (int, int)
        Kích thước figure.
    assume_bgr : bool
        True nếu ảnh màu là BGR (cv2); sẽ auto chuyển RGB để vẽ.
    """
    n = len(images)
    rows = math.ceil(n / cols)
    plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        ax = plt.subplot(rows, cols, i + 1)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(bgr_to_rgb(img) if assume_bgr else img)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_with_cv2(img: np.ndarray, window_name: str = "Image",
                  delay: int = 0, destroy: bool = True) -> None:
    """
    Hiển thị ảnh bằng OpenCV (mở cửa sổ riêng).

    Parameters
    ----------
    delay : int
        Thời gian chờ phím (ms). 0 = chờ vô hạn.
    destroy : bool
        True → đóng tất cả cửa sổ sau khi chờ phím.
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(delay)
    if destroy:
        cv2.destroyAllWindows()
