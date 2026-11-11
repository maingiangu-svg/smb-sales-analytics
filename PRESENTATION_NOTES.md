# Gợi ý thuyết trình phiên bản bán lẻ

## Câu hỏi của nhóm

1. Khu vực và nhóm sản phẩm nào tạo doanh thu nhưng lợi nhuận thấp hoặc âm?
2. Chiết khấu liên hệ thế nào với biên lợi nhuận sau khi xét danh mục, khu vực và năm?
3. Dự báo doanh thu theo ngày bằng Holt-Winters có tốt hơn cách dự báo cùng thứ tuần trước không?

## Nguồn và kiểm toán

- Bản `archive.zip` thầy gợi ý chỉ có 9.800 dòng và 18 cột; không có `Profit`, `Discount`.
- Nhóm dùng bản Sample Superstore đầy đủ 21 cột, 9.994 dòng, từ 03/01/2014 đến 30/12/2017. Nguồn: <https://github.com/Himanshi252005/Superstore-Sales-Analysis/blob/main/Sample%20-%20Superstore.csv>.
- Kiểm toán: 0 ô thiếu, 0 Row ID trùng, 0 dòng bị loại bởi quy tắc hợp lệ. `Sales`, `Profit`, `Discount` là cột nguồn; không mô phỏng.
- Mỗi dòng là một mặt hàng trong đơn. Tổng số đơn riêng biệt là 5.009, không phải 9.994.

## Kết quả khi chưa lọc

| Chỉ số | Kết quả |
| --- | ---: |
| Doanh thu | $2,297,200.86 |
| Lợi nhuận | $286,397.02 |
| Biên lợi nhuận tổng | 12.47% |
| Khu vực có lợi nhuận cao nhất | West: $108,418.45 |
| Khu vực có biên lợi nhuận thấp nhất | Central: 7.92% |

Nhóm chiết khấu trên 20% có tổng lợi nhuận âm trong mẫu. Hồi quy OLS trên biên lợi nhuận, kiểm soát log doanh thu, số lượng, danh mục, khu vực và năm, ước lượng mức chiết khấu tăng 10 điểm phần trăm đi cùng biên lợi nhuận thấp hơn khoảng 19,4 điểm phần trăm (CI 95%: -19,8 đến -19,0). Đây là liên hệ thống kê trong dữ liệu quan sát, không phải kết quả A/B và không chứng minh tác động nhân quả. Không nhân hệ số này với toàn bộ doanh thu để tuyên bố “tiết kiệm được” một khoản tiền.

## Dự báo và giới hạn

- Chuỗi có 1.458 ngày lịch từ 03/01/2014 đến 30/12/2017, trong đó 221 ngày không có giao dịch ghi nhận được gán doanh thu 0. Nguồn không cho biết cửa hàng đóng cửa hay chỉ không phát sinh đơn.
- Holt-Winters dùng mùa vụ 7 ngày. Kiểm tra ở 3 khoảng 14 ngày liên tiếp cuối lịch sử: MAE của Holt-Winters khoảng $1.921/ngày, của cách dự báo cùng thứ tuần trước khoảng $3.034/ngày. WAPE lần lượt khoảng 63,5% và 100,3%; sai số vẫn lớn, nên chỉ trình bày như mô hình thử nghiệm.
- Đây là dữ liệu mẫu công khai cũ. Dự báo các ngày sau 30/12/2017 là minh họa phương pháp, không phải dự báo cho thời điểm hiện tại.

## Sửa slide cũ trước khi trình bày

PDF cũ nói về quán cà phê và doanh thu theo tháng; không nên dùng để trình bày phiên bản bán lẻ. Thay biểu đồ tháng, MAE/RMSE tháng và tuyên bố về chi phí quán cà phê bằng số liệu mới. Chỉ nói các bước làm sạch, chuẩn hóa hoặc A/B test nếu có mã và kết quả thật tương ứng.
