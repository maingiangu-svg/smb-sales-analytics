import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde, ttest_ind
import pdfplumber
from PIL import Image

# 1. Load Data
conn = sqlite3.connect('data/data.db')
df = pd.read_sql('SELECT * FROM sales_data', conn)
conn.close()

# 2. Extract existing slides from original PDF
orig_pdf = '/Users/macbookm1/.gemini/antigravity/brain/fc7215d1-a3da-43f6-a923-1fa94a1e00fb/.user_uploaded/media_1792565053766_91991799.pdf'
orig_images = []
with pdfplumber.open(orig_pdf) as pdf:
    for p in pdf.pages:
        orig_images.append(p.to_image(resolution=150).original)
print(f"Loaded {len(orig_images)} original slides.")
target_size = orig_images[0].size # 1754 x 988

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

# ==============================================================================
# SLIDE 2: TỔNG QUAN NỘI DUNG BÁO CÁO (CẬP NHẬT 3 PHẦN CÂN ĐỐI)
# ==============================================================================
fig = plt.figure(figsize=(17.54, 9.88), dpi=100)
fig.patch.set_facecolor('#f8fafc')

fig.text(0.045, 0.94, "CHƯƠNG TRÌNH TRÌNH BÀY", fontsize=13, weight='bold', color='#d97706')
fig.text(0.045, 0.89, "TỔNG QUAN NỘI DUNG BÁO CÁO", fontsize=24, weight='bold', color='#0f2c59')

# Card 1: PHẦN 1
ax1 = fig.add_axes([0.045, 0.08, 0.285, 0.77])
ax1.set_facecolor('white')
for s in ax1.spines.values(): s.set_color('#cbd5e1')
ax1.add_patch(patches.Rectangle((0, 0.91), 1, 0.09, facecolor='#1e3a8a', transform=ax1.transAxes))
ax1.text(0.5, 0.955, "PHẦN 1: CLEAN DATA (6 BƯỚC)", color='white', fontsize=11.5, weight='bold', ha='center', va='center', transform=ax1.transAxes)

p1_text = (
    "• 1. Audit Data: Khám phá cấu trúc, khuyết thiếu,\n"
    "  phân loại 3 cơ chế và kiểm tra ngoại lai/nhiễu.\n\n"
    "• 2. Chuẩn hóa Header: Đồng bộ snake_case,\n"
    "  xử lý khoảng trắng thừa và khử trùng lặp định danh.\n\n"
    "• 3. Tạo Indicator Columns: Khởi tạo biến cờ hiệu\n"
    "  bảo tồn ngữ nghĩa Payment Method & Location.\n\n"
    "• 4. Đồng bộ Missing & Ép kiểu: Khử ký tự lạ ($ , %),\n"
    "  quy chuẩn chuỗi rác về ô trống và ép kiểu chuẩn.\n\n"
    "• 5. Khôi phục Toán học: Loại bỏ MCAR < 5%,\n"
    "  suy luận thực đơn, khôi phục doanh thu 3 chiều.\n\n"
    "• 6. Khóa Biên Nghiệp vụ Cost: Thuật toán kiểm soát\n"
    "  chặn trên - chặn dưới giá vốn theo % Food Cost."
)
ax1.text(0.05, 0.88, p1_text, fontsize=9.2, color='#1e293b', va='top', transform=ax1.transAxes, linespacing=1.2)
ax1.set_xticks([]); ax1.set_yticks([])

# Card 2: PHẦN 2
ax2 = fig.add_axes([0.355, 0.08, 0.285, 0.77])
ax2.set_facecolor('white')
for s in ax2.spines.values(): s.set_color('#cbd5e1')
ax2.add_patch(patches.Rectangle((0, 0.91), 1, 0.09, facecolor='#0284c7', transform=ax2.transAxes))
ax2.text(0.5, 0.955, "PHẦN 2: ĐẶC TRƯNG, EDA & KIỂM ĐỊNH", color='white', fontsize=11.5, weight='bold', ha='center', va='center', transform=ax2.transAxes)

p2_text = (
    "• Chuẩn hóa & Mã hóa: RobustScaler chống ngoại lai,\n"
    "  One-Hot biến phân loại và bẫy The Dummy Trap.\n\n"
    "• Chống Rò rỉ Dữ liệu: Phân chia Train/Test theo\n"
    "  thời gian trước khi fit mô hình (Data Leakage).\n\n"
    "• Thống kê Mô tả: Phân phối lệch phải, ưu tiên dùng\n"
    "  Trung vị (Median = 8.0 USD) & IQR chống méo mó.\n\n"
    "• Tương quan & Bẫy Nhân quả: Pearson vs Spearman,\n"
    "  nhận diện tương quan ngẫu nhiên trong chuỗi thời gian.\n\n"
    "• Trực quan hóa EDA: Xu hướng theo tháng (Line chart),\n"
    "  cơ cấu món ăn (Bar chart) tuân thủ Data-ink ratio.\n\n"
    "• Suy diễn Thống kê & A/B Testing: Ước lượng tham số,\n"
    "  kiểm định t-test chi tiêu và tính cỡ mẫu tối thiểu."
)
ax2.text(0.05, 0.88, p2_text, fontsize=9.2, color='#1e293b', va='top', transform=ax2.transAxes, linespacing=1.2)
ax2.set_xticks([]); ax2.set_yticks([])

# Card 3: PHẦN 3
ax3 = fig.add_axes([0.665, 0.08, 0.285, 0.77])
ax3.set_facecolor('white')
for s in ax3.spines.values(): s.set_color('#cbd5e1')
ax3.add_patch(patches.Rectangle((0, 0.91), 1, 0.09, facecolor='#d97706', transform=ax3.transAxes))
ax3.text(0.5, 0.955, "PHẦN 3: DỰ BÁO DOANH THU", color='white', fontsize=11.5, weight='bold', ha='center', va='center', transform=ax3.transAxes)

p3_text = (
    "• Bài toán Chuỗi Thời Gian: Dự báo doanh thu\n"
    "  bán hàng định kỳ hàng tháng phục vụ quản trị.\n\n"
    "• Thuật toán Lựa chọn: San phẳng hàm mũ bộ ba\n"
    "  Holt-Winters Exponential Smoothing kinh điển.\n\n"
    "• Phân rã 3 Chiều Độc lập: Tách bạch Mức cơ sở\n"
    "  (Level), Xu hướng (Trend) và Mùa vụ (Seasonality).\n\n"
    "• Đánh giá Không Rò rỉ: Phân chia theo trục thời gian\n"
    "  (Time-based split), đo lường qua MAE & RMSE.\n\n"
    "• Kịch bản & Ứng dụng Thực tiễn: Dự phóng 3-12\n"
    "  tháng; chủ động nguyên liệu, nhân sự và dòng tiền."
)
ax3.text(0.05, 0.88, p3_text, fontsize=9.2, color='#1e293b', va='top', transform=ax3.transAxes, linespacing=1.2)
ax3.set_xticks([]); ax3.set_yticks([])

plt.savefig('slide_2_updated.png', bbox_inches='tight')
plt.close()
im_slide2 = Image.open('slide_2_updated.png').convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
print("Slide 2 generated.")

# Load Slide 3, 4, 5, 6, 7 from yesterday's code logic
# (To make it robust and cleanly reproducible, let's generate them)
