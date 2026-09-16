# 📊 SMB Sales Analytics & Pricing Optimization Platform

Một nền tảng phân tích dữ liệu bán lẻ và gợi ý mức giá/chiết khấu tối ưu dành cho doanh nghiệp vừa và nhỏ (SMB). Dự án bao gồm Data Pipeline làm sạch dữ liệu tự động, lưu trữ SQLite, Dashboard tương tác trên Streamlit và mô hình Machine Learning dự báo lợi nhuận.

---

## 🛠️ Kiến Trúc Hệ Thống

```text
Local Raw Data (CSV/Excel)
       │
       ▼
Data Pipeline (src/pipeline.py) ──► Regex Cleaning & Feature Engineering
       │
       ▼
SQLite Database (data/data.db)
       │
       ├──► Streamlit Dashboard (app.py & src/eda.py)
       └──► ML Pricing Model (src/model.py)
🚀 Tính Năng Chính
Data Pipeline (ETL): Tự động đọc file dữ liệu local, chuẩn hóa tên cột SQL-friendly, làm sạch ký tự tiền tệ bẩn ($, ,, %) và bóc tách thời gian (year, month, day_of_week).

Interactive Dashboard:

Hiển thị chỉ số KPI chính (Doanh thu, Lợi nhuận, Số đơn hàng, Chiết khấu trung bình).

Biểu đồ xu hướng doanh số theo thời gian và hiệu năng theo danh mục sản phẩm.

Bộ lọc tương tác đa chiều trên Sidebar.

Pricing Optimization (ML): Huấn luyện mô hình hồi quy dự báo Lợi nhuận dựa trên Chiết khấu và Doanh số, gợi ý mức Chiết khấu (Discount) tối ưu.

📁 Cấu Trúc Thư Mục
Plaintext
smb-sales-analytics/
├── data/
│   ├── train.csv           # File dữ liệu thô mẫu
│   └── data.db             # Database SQLite (tự động khởi tạo)
├── src/
│   ├── pipeline.py         # Pipeline làm sạch và nạp dữ liệu
│   ├── eda.py              # Các hàm trực quan hóa dữ liệu (Plotly)
│   └── model.py            # Module Machine Learning & Optimization
├── app.py                  # Giao diện chính Streamlit
├── requirements.txt        # Danh sách thư viện phụ thuộc
└── README.md
⚙️ Hướng Dẫn Cài Đặt & Chạy Dự Án
1. Cài đặt môi trường
Bash
# Clone repository
git clone [https://github.com/maingiangu-svg/smb-sales-analytics.git](https://github.com/maingiangu-svg/smb-sales-analytics.git)
cd smb-sales-analytics

# Tạo và kích hoạt môi trường ảo (tùy chọn)
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
2. Chạy Data Pipeline
Nạp dữ liệu từ file thô vào Database SQLite:

Bash
python -m src.pipeline
3. Khởi chạy Web Dashboard
Bash
streamlit run app.py
Giao diện sẽ tự động mở tại địa chỉ http://localhost:8501.