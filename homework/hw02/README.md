"hw02/\n"
"│\n"
"├── data/\n"
"│ └── bin/ # Chứa file ảnh .bin đầu vào\n"
"│\n"
"├── results/ # Thư mục kết quả (ảnh .png, file docx, ...)\n"
"│\n"
"├── src/\n"
"│ ├── task_01.py # Task 1: Thresholding & Contour\n"
"│ ├── task_02.py # Task 2: Contrast Stretching\n"
"│ ├── task_03.py # Task 3: Binary Template Matching\n"
"│ ├── task_04.py # Task 4: Histogram Equalization\n"
"│ │\n"
"│ └── utils/\n"
"│ ├── io_utils.py # Đọc/ghi ảnh nhị phân, ảnh màu\n"
"│ └── display_utils.py # Hàm hiển thị ảnh bằng matplotlib / cv2\n"
"│\n"
"└── README.md"

Task 1: Đọc và hiển thị ảnh nhị phân (.bin)
Yêu cầu: Đọc dữ liệu ảnh từ file nhị phân (.bin) và hiển thị dưới dạng ảnh xám.
Cách giải quyết:

- Đọc file nhị phân với kích thước ảnh cho trước (width, height).
- Chuyển dữ liệu thô thành mảng numpy 2D.
- Sử dụng matplotlib để hiển thị ảnh.
  Kết quả: Ảnh được hiển thị đúng kích thước và giá trị pixel.
  Task 2: Thực hiện Linear Contrast Stretching
  Yêu cầu: Cải thiện độ tương phản ảnh bằng phương pháp Linear Contrast Stretching.
  Cách giải quyết:
- Tính giá trị pixel nhỏ nhất (min) và lớn nhất (max) trong ảnh.
- Áp dụng công thức: new_pixel = (pixel - min) \* 255 / (max - min).
- Hiển thị ảnh trước và sau khi xử lý.
  Kết quả: Ảnh sau khi kéo dãn độ tương phản rõ nét hơn (tăng sự phân bố giá trị pixel).
  Task 3: Vẽ Histogram của ảnh
  Yêu cầu: Vẽ biểu đồ histogram biểu diễn phân bố mức xám trong ảnh.
  Cách giải quyết:
- Dùng numpy để đếm tần suất xuất hiện của các giá trị pixel (0-255).
- Sử dụng matplotlib để vẽ biểu đồ histogram.
  Kết quả: Histogram cho thấy sự phân bố sáng tối trong ảnh, hỗ trợ phân tích chất lượng ảnh.
  Task 4: Histogram Equalization
  Yêu cầu: Cân bằng histogram để cải thiện chất lượng ảnh.
  Cách giải quyết:
- Tính histogram của ảnh.
- Tính hàm phân phối tích lũy (CDF).
- Chuẩn hóa CDF để trải đều giá trị pixel từ 0 đến 255.
- Áp dụng công thức để tạo ảnh mới.
- Hiển thị cả ảnh gốc và ảnh đã cân bằng histogram.
  Kết quả: Ảnh sau khi cân bằng histogram có độ tương phản tốt hơn, chi tiết vùng sáng/tối được cải thiện.
