# Bản thử nghiệm quán cà phê (lưu trữ)

Đây là dữ liệu, notebook, mô hình và mã phân tích trước khi nhóm chuyển sang chủ đề bán lẻ Superstore. Các file được chuyển nguyên nội dung để có thể xem lại lịch sử; dashboard ở thư mục gốc hiện không dùng chúng.

- `data/`: dữ liệu thô, dữ liệu đã xử lý, SQLite và mô hình dự báo cũ.
- `src/`: pipeline, EDA, thống kê, báo cáo và mô hình cũ.
- `baseline.ipynb`, `advanced_model.ipynb`: notebook của bản thử nghiệm.
- `slide_experiments/`: các script tạo/thử giao diện slide. Một số script tham chiếu file PDF ngoài dự án nên không chạy độc lập từ kho lưu trữ này.

Nếu cần chạy lại notebook cũ, mở nó với thư mục làm việc là `archive/cafe` và cài thêm các thư viện trong `requirements.txt` của thư mục này. Các số liệu hoặc kết luận trong bản cũ không dùng cho báo cáo Superstore hiện tại.
