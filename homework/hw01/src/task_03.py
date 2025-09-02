# ==========================================================
# Task 3: Color band swapping
# - (a) Read lena512color.jpg (512x512, 24-bit RGB)
# - (b) Display original
# - (c) Create J2 by swapping color bands:
#       Red <- Blue, Green <- Red, Blue <- Green
# - (d) Display & save result
# ==========================================================

from utils.io_utils import read_image, split_bgr, merge_bgr, write_image
from utils.display_utils import show_images_grid

def task3():
    # (a) Đọc ảnh màu JPG
    J1 = read_image("homework/hw01/data/img/lena512color.jpg", as_gray=False)  # BGR

    # (b,c) Tách kênh và hoán đổi
    b, g, r = split_bgr(J1)
    J2 = merge_bgr(r, b, g)  # Red<-Blue, Green<-Red, Blue<-Green

    # (d) Hiển thị và ghi ra file
    show_images_grid([J1, J2], ["Original (J1)", "Band-swapped (J2)"])
    write_image("homework/hw01/results/task3_lena512color_swapped.jpg", J2)


if __name__ == "__main__":
    task3()
