# ==========================================================
# Task 2: Negative transformation on grayscale JPEG
# - (a) Read lenagray.jpg
# - (b) Compute photographic negative J2 = 255 - J1
# - (c) Display and save result
# ==========================================================

from utils.io_utils import read_image, ensure_uint8, write_image
from utils.display_utils import show_images_grid

def task2():
    # (a) Đọc ảnh grayscale JPG
    J1 = read_image("homework/hw01/data/img/lenagray.jpg", as_gray=True)

    # (b) Negative
    J2 = ensure_uint8(255 - J1)

    # (c) Hiển thị và ghi ra file
    show_images_grid([J1, J2], ["Original (J1)", "Negative (J2)"])
    write_image("homework/hw01/results/task2_lenagray_negative.jpg", J2)


if __name__ == "__main__":
    task2()
