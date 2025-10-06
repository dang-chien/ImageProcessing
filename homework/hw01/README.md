📂 Cấu trúc thư mục
hw01/
│── data/ # Dữ liệu gốc
│ ├── bin/ # File ảnh dạng nhị phân
│ │ ├── lena.bin
│ │ └── peppers.bin
│ ├── img/ # Ảnh gốc (.jpg)
│ │ ├── lena512color.jpg
│ │ └── lenagray.jpg
│ └── docs/ # Tài liệu tham khảo
│ ├── hw-01.pdf
│ └── processing.txt
│
│── results/ # Kết quả xử lý
│ ├── hw01.doc # Hình chụp các kết quả thu được
│ ├── task1_J.png
│ ├── task1_K.png
│ ├── task2_lenagray_negative.jpg
│ └── task3_lena512color_swapped.jpg
│
│── src/ # Mã nguồn chính
│ ├── utils/ # Thư viện tiện ích
│ │ ├── **pycache**/
│ │ ├── display_utils.py # Hàm hỗ trợ hiển thị ảnh
│ │ └── io_utils.py # Hàm đọc/ghi file ảnh
│ ├── task_01.py # Bài tập 1
│ ├── task_02.py # Bài tập 2
│ └── task_03.py # Bài tập 3
│
└── README.md # File mô tả (tài liệu này)

🚀 Cách chạy chương trình

Chạy từng bài tập

Bài tập 1:

python src/task_01.py

Bài tập 2:

python src/task_02.py

Bài tập 3:

python src/task_03.py

Kết quả
Các ảnh kết quả sẽ được lưu trong thư mục results/ với tên tương ứng:

task1_J.png, task1_K.png

task2_lenagray_negative.jpg

task3_lena512color_swapped.jpg

📖 Nội dung các bài tập

Task 1: Xử lý cơ bản ảnh nhị phân và hiển thị dưới dạng ảnh xám.

Task 2: Tạo ảnh âm bản (negative image) từ ảnh grayscale.

Task 3: Thay đổi thứ tự kênh màu RGB của ảnh màu.
