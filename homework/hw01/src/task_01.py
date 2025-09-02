# 1.  Obtain the images “lena.bin” and “peppers.bin” from the dataset  Each image has 256 × 256 pixels and each pixel has 8 bits. 
# (a)  Read and display the images. 
# Quy trình: file nhị phân → numpy array → reshape → ánh xạ colormap → display.
# Thêm các thư viện xử lý số 
# numpy Libary: Xử lý mảng số e
# matplotlib: hiện thị ảnh thích hợp với hiện thị ảnh trực quan và phụ vụ cho khoa học dữ liệu
# OpenCV (cv2): cũng hiện thị ảnh nhưng phù hợp cho các ứng dụng real-time: camera, xử lý ảnh động

import numpy as np
import matplotlib.pyplot as plt # hiện thị ảnh
import cv2

#1: Chúng ta sẽ đọc file .bin
f = open('homework/hw01/rawBinary/lena.bin', 'rb')
img_bin = f.read()  #Lúc này file .bin của ảnh sẽ là một chuỗi nhị phân (byte) - nhưng chưa phải là mảng NUMPY
# print(len(bin))

#2: Bây giờ chúng ta sẽ chuyển nó sang mảng NUMPY 1D

img = np.frombuffer(img_bin,dtype=np.uint8) # ==> Lúc này mảng img chính thức là mảng NUMPY 1D với lenght = số pixel

#3: Bây giờ chúng ta sẽ reshape nó sang dạng 2D (256 x 256)

img = img.reshape((256,256))

#4.1 Giờ chúng ta sẽ sử dụng thư viện matploplib.pyplot để hiện thị ảnh từ  mảng 2 chiều
    # plt.imshow(img, cmap="gray")  # cmap="gray" để mapping 0–255 thành mức xám
    # plt.title("Lena")
    # plt.axis("off")
    # plt.show()

#4.2 Giờ chúng ta sẽ sử dụng thư viện OpenCV để hiện thị ảnh từ  mảng 2 chiều
cv2.imshow("Lena", img) # Hiển thị ảnh GrayScale
cv2.waitKey(0)
cv2.destroyAllWindows()












# (b)  Define  a  new  256  256  image  J  as  follows:  the  left  half  of  J,  i.e.,  the  first  128 columns,  should  be  equal  to  the  left  half  of  the  Lena  image.  The  right  half  of 
# J,  i.e.,  the  129th  column  through  the  256th  column,  should  be  equal  to  the  right half of the Peppers  image. 
# (c)  Define a new 256 × 256 image K by swapping the left and right halves of J. 
# (d)  Be  sure  to  turn  in:  A  listing  of  your  code  and  printouts  of  the  original  images, image J,  and  image  K.

