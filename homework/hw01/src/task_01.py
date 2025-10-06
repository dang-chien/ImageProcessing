# ==========================================================
# Task 1: Image composition with RAW binary grayscale images
# - (a) Read & display lena.bin, peppers.bin (256x256, 8-bit)
# - (b) Build J: left half = Lena, right half = Peppers
# - (c) Build K: swap left-right halves of J
# - (d) Save and display results
# ==========================================================

from utils.io_utils import read_bin_image, write_image
from utils.display_utils import show_images_grid

import os
print("Current working directory:", os.getcwd())


def task1():
    # (a) Đọc ảnh lena & peppers
    lena = read_bin_image("homework/hw01/data/bin/lena.bin", width=256, height=256)
    peppers = read_bin_image("homework/hw01/data/bin/peppers.bin", width=256, height=256)

    # (b) Tạo ảnh J
    J = lena.copy()
    J[:, 128:] = peppers[:, 128:]

    # (c) Tạo ảnh K (swap left-right halves của J)
    K = J.copy()
    K[:, :128], K[:, 128:] = J[:, 128:], J[:, :128]

    # (d) Hiển thị & lưu ảnh
    show_images_grid(
        [lena, peppers, J, K],
        ["Lena", "Peppers", "J (Lena|Peppers)", "K (Swap halves)"],
        cols=2
    )
    write_image("homework/hw01/results/task1_J.png", J)
    write_image("homework/hw01/results/task1_K.png", K)


if __name__ == "__main__":
    task1()
