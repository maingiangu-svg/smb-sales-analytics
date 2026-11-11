# Phân tích doanh thu và lợi nhuận bán lẻ

Đồ án môn Nhập môn Khoa học dữ liệu: kiểm toán dữ liệu giao dịch Superstore, phân tích doanh thu/lợi nhuận theo khu vực và sản phẩm, khảo sát mối liên hệ giữa chiết khấu và biên lợi nhuận, và dự báo doanh thu theo ngày.

## Nguồn dữ liệu

- `data/superstore_raw.csv`: bản Sample Superstore 21 cột, 9.994 dòng, 2014-01-03 đến 2017-12-30. Tải từ [bản công khai trên GitHub](https://github.com/Himanshi252005/Superstore-Sales-Analysis/blob/main/Sample%20-%20Superstore.csv), cùng họ dữ liệu với [Sample Superstore trên Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final). Tệp gốc được giữ nguyên, mã hóa Latin-1.
- `archive.zip` mà nhóm nhận được là bản Superstore `train.csv` 9.800 dòng, 18 cột, chỉ có `Sales`. Bản này không đủ để phân tích lợi nhuận hoặc chiết khấu như yêu cầu đề tài.
- Dữ liệu quán cà phê và các notebook cũ nằm trong `archive/cafe/`. Dashboard hiện tại đọc `data/retail.db`, được tạo lại bằng pipeline và không cần đưa vào Git.

Sample Superstore là dữ liệu học tập công khai, không đại diện cho sổ sách của một doanh nghiệp thật. Không công bố tên khách hàng từ nguồn trong giao diện.

## Chạy dự án

```bash
python3 -m pip install -r requirements.txt
python3 -m src.retail_pipeline
streamlit run app.py
```

Pipeline chuẩn hóa tên cột, kiểm tra định dạng ngày, khoảng chiết khấu và giá trị số, rồi lưu dữ liệu hợp lệ vào SQLite. `profit` và `discount` là giá trị quan sát trong CSV, không được tạo bằng công thức hay điền giả. File nguồn có 0 ô thiếu và 0 Row ID trùng; sau kiểm tra còn 9.994 dòng.

## Phương pháp

1. **EDA:** tổng doanh thu, lợi nhuận, biên lợi nhuận (`sum(profit) / sum(sales)`), số đơn duy nhất; đối chiếu khu vực và nhóm sản phẩm lỗ.
2. **Chiết khấu:** thống kê theo các mức 0%, 0–20%, 20–40%, 40–60%, >60%. Hồi quy OLS của biên lợi nhuận theo chiết khấu, log doanh thu, số lượng, danh mục, khu vực, năm; sai số chuẩn HC3. Hệ số chiết khấu là *mối liên hệ có điều kiện*, không phải tác động nhân quả. Đây không phải A/B test.
3. **Dự báo:** cộng doanh thu theo ngày, lấp ngày không có giao dịch ghi nhận bằng 0, dùng Holt-Winters với chu kỳ tuần 7 ngày. So sánh với dự báo “cùng thứ tuần trước” trên 3 khoảng kiểm tra liên tiếp, mỗi khoảng dài bằng chân trời dự báo. Báo MAE và WAPE; chỉ dùng lịch sử có trước từng khoảng kiểm tra. Sau đó fit trên toàn bộ lịch sử và dự báo 7–28 ngày.

Lưu ý: 0 vào ngày không có giao dịch là quy ước phân tích; nguồn không cho biết ngày cửa hàng đóng cửa. Dự báo chỉ nên được diễn giải như thử nghiệm trên bộ dữ liệu mẫu. Biểu đồ/hệ số có thể thay đổi khi lọc khu vực hoặc danh mục.

## Cấu trúc chính

- `src/retail_pipeline.py`: đọc, kiểm toán, làm sạch và lưu SQLite.
- `src/retail_analysis.py`: tổng hợp lợi nhuận và hồi quy.
- `src/retail_forecast.py`: chuỗi ngày, kiểm tra theo thời gian và dự báo.
- `app.py`: dashboard Streamlit.

Các notebook, mô hình và script slide quán cà phê thuộc phiên bản cũ nằm trong `archive/cafe/`; số liệu của chúng không mô tả pipeline bán lẻ hiện tại.
