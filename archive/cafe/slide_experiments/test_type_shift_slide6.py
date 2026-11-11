import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde

conn = sqlite3.connect('data/data.db')
df = pd.read_sql('SELECT * FROM sales_data', conn)
conn.close()

fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

# Type Shift smoky aura background
ax_bg = fig.add_axes([0, 0, 1, 1])
ax_bg.set_facecolor('#f8f9fa')
ax_bg.axis('off')
x = np.linspace(-1, 1, 200)
y = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(x, y)
Z = 1.0 - 0.12 * (X**2 + Y**2)
ax_bg.imshow(Z, cmap='Greys_r', extent=[0, 1, 0, 1], alpha=0.08, origin='lower', aspect='auto')

# Header
fig.text(0.05, 0.93, "DATA-DRIVEN PROJECT  //  PHẦN 2", fontsize=11, fontfamily='monospace', weight='bold', color='#2547eb')
fig.text(0.05, 0.86, "DISTRIBUTION & DESCRIPTIVE STATS", fontsize=32, fontfamily='sans-serif', weight='bold', color='#0f172a')
fig.text(0.05, 0.825, "Nguyên tắc Shape picks Statistic: Phân phối doanh thu lệch phải và thước đo trung tâm bền vững.", fontsize=13, fontfamily='sans-serif', color='#475569')
fig.text(0.95, 0.93, "[ 06 / 14 ]", fontsize=12, fontfamily='monospace', weight='bold', color='#2547eb', ha='right')

# Divider hairline
ax_line = fig.add_axes([0.05, 0.80, 0.90, 0.002])
ax_line.set_facecolor('#0f172a')
ax_line.axis('off')

# ==============================================================================
# LEFT: 3 STATISTICAL CARDS
# ==============================================================================
# Stat Card 1: MEDIAN (King of Location)
ax_s1 = fig.add_axes([0.05, 0.52, 0.42, 0.24])
ax_s1.set_facecolor('white')
for s in ax_s1.spines.values(): s.set_color('#0f172a'); s.set_linewidth(1.2)
ax_s1.set_xticks([]); ax_s1.set_yticks([])
ax_s1.text(0.05, 0.82, "[ THƯỚC ĐO VỊ TRÍ ]", fontfamily='monospace', fontsize=9.5, weight='bold', color='#2547eb', transform=ax_s1.transAxes)
ax_s1.text(0.05, 0.48, "MEDIAN = 8.0 USD", fontfamily='sans-serif', fontsize=22, weight='bold', color='#0f172a', transform=ax_s1.transAxes)
ax_s1.text(0.05, 0.20, "• Bác bỏ Mean = 8.92 USD (bị kéo lệch ảo bởi đơn lớn)\n• Phản ánh trung thực: 50% khách hàng chi tiêu <= 8.0 USD", fontfamily='sans-serif', fontsize=10.5, color='#334155', transform=ax_s1.transAxes, linespacing=1.3)

# Stat Card 2: IQR (Spread)
ax_s2 = fig.add_axes([0.05, 0.26, 0.42, 0.23])
ax_s2.set_facecolor('white')
for s in ax_s2.spines.values(): s.set_color('#0f172a'); s.set_linewidth(1.2)
ax_s2.set_xticks([]); ax_s2.set_yticks([])
ax_s2.text(0.05, 0.82, "[ THƯỚC ĐO ĐỘ PHÂN TÁN ]", fontfamily='monospace', fontsize=9.5, weight='bold', color='#2547eb', transform=ax_s2.transAxes)
ax_s2.text(0.05, 0.48, "IQR = 8.0 USD (Q1: $4 - Q3: $12)", fontfamily='sans-serif', fontsize=20, weight='bold', color='#0f172a', transform=ax_s2.transAxes)
ax_s2.text(0.05, 0.20, "• Thay thế Độ lệch chuẩn Std = 6.0 USD vì nhạy cảm ngoại lai\n• 50% lượng đơn hàng trọng tâm của quán nằm gọn ở $4.0 - $12.0", fontfamily='sans-serif', fontsize=10.5, color='#334155', transform=ax_s2.transAxes, linespacing=1.3)

# Stat Card 3: SKEWNESS (Shape)
ax_s3 = fig.add_axes([0.05, 0.10, 0.42, 0.13])
ax_s3.set_facecolor('#0f172a')
for s in ax_s3.spines.values(): s.set_color('#0f172a')
ax_s3.set_xticks([]); ax_s3.set_yticks([])
ax_s3.text(0.05, 0.65, "[ HÌNH DẠNG ]  SKEWNESS = +0.83 > 0  (LỆCH PHẢI)", fontfamily='monospace', fontsize=11, weight='bold', color='#38bdf8', transform=ax_s3.transAxes)
ax_s3.text(0.05, 0.25, "Đuôi dài kéo về phía doanh thu cao. Bắt buộc dùng Median & RobustScaler.", fontfamily='sans-serif', fontsize=10, color='#e2e8f0', transform=ax_s3.transAxes)

# ==============================================================================
# RIGHT: MODERN CHART (Sharp 1px Border Box)
# ==============================================================================
ax_chart_box = fig.add_axes([0.50, 0.10, 0.45, 0.66])
ax_chart_box.set_facecolor('white')
for s in ax_chart_box.spines.values(): s.set_color('#0f172a'); s.set_linewidth(1.2)
ax_chart_box.set_xticks([]); ax_chart_box.set_yticks([])

ax_chart_box.text(0.06, 0.94, "[ VISUAL PROOF ]", fontfamily='monospace', fontsize=10, weight='bold', color='#2547eb', transform=ax_chart_box.transAxes)
ax_chart_box.text(0.06, 0.88, "BIỂU ĐỒ PHÂN PHỐI DOANH THU ĐƠN HÀNG", fontfamily='sans-serif', fontsize=16, weight='bold', color='#0f172a', transform=ax_chart_box.transAxes)
ax_chart_box.text(0.06, 0.83, "Vạch đỏ (Median) đứng trước vạch vàng (Mean), minh chứng thực nghiệm độ lệch phải.", fontfamily='sans-serif', fontsize=11, color='#64748b', transform=ax_chart_box.transAxes)

# Embed plot inside right box
ax_p = fig.add_axes([0.54, 0.16, 0.38, 0.45])
ax_p.set_facecolor('#f8fafc')
n, bins, _ = ax_p.hist(df['sales'], bins=15, color='#2547eb', edgecolor='white', linewidth=1.2, alpha=0.85, density=True)
density = gaussian_kde(df['sales'])
xs = np.linspace(df['sales'].min(), df['sales'].max(), 200)
ax_p.plot(xs, density(xs), color='#0f172a', linewidth=2.5, label='KDE (Đường mật độ)')
ax_p.axvline(8.0, color='#dc2626', linestyle='--', linewidth=2.2, label='Median = 8.0 USD')
ax_p.axvline(8.92, color='#d97706', linestyle=':', linewidth=2.2, label='Mean = 8.92 USD')
ax_p.set_xlabel("Doanh thu mỗi đơn hàng (USD)", fontsize=11, fontfamily='sans-serif')
ax_p.set_ylabel("Mật độ tần suất", fontsize=11, fontfamily='sans-serif')
ax_p.legend(fontsize=10, loc='upper right', frameon=True, edgecolor='#0f172a')
ax_p.grid(axis='y', linestyle=':', alpha=0.5)
ax_p.spines[['top', 'right']].set_visible(False)
ax_p.spines[['left', 'bottom']].set_color('#0f172a')

plt.savefig('/tmp/sample_type_shift_slide6.png', bbox_inches='tight', dpi=150)
plt.close()
print("Sample Type Shift Slide 6 created successfully.")
