import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter

# Canvas size 1920x1080 (Full HD 16:9)
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

# Create subtle Type Shift background: warm/cool off-white with dark smoky edges
ax_bg = fig.add_axes([0, 0, 1, 1])
ax_bg.set_facecolor('#f8f9fa')
ax_bg.axis('off')

# Gradient aura mesh (simulating smoky gradient)
x = np.linspace(-1, 1, 200)
y = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(x, y)
# Darker corners / edges
Z = 1.0 - 0.12 * (X**2 + Y**2)
ax_bg.imshow(Z, cmap='Greys_r', extent=[0, 1, 0, 1], alpha=0.08, origin='lower', aspect='auto')

# Top header: Minimalist editorial style
fig.text(0.05, 0.93, "DATA-DRIVEN PROJECT  //  PHẦN 2", fontsize=11, fontfamily='monospace', weight='bold', color='#2547eb')
fig.text(0.05, 0.86, "TRANSFORMATION & SCALING", fontsize=32, fontfamily='sans-serif', weight='black', color='#0f172a')
fig.text(0.05, 0.825, "Chuẩn hóa thang đo, mã hóa danh mục và nguyên tắc độc lập chống rò rỉ dữ liệu.", fontsize=13, fontfamily='sans-serif', color='#475569')

# Top right badge
fig.text(0.95, 0.93, "[ 05 / 14 ]", fontsize=12, fontfamily='monospace', weight='bold', color='#2547eb', ha='right')

# Divider hairline
ax_line = fig.add_axes([0.05, 0.80, 0.90, 0.002])
ax_line.set_facecolor('#0f172a')
ax_line.axis('off')

# ==============================================================================
# CARD 1: ROBUST SCALER (Sharp 1px border box)
# ==============================================================================
ax1 = fig.add_axes([0.05, 0.10, 0.28, 0.66])
ax1.set_facecolor('white')
for s in ax1.spines.values():
    s.set_color('#0f172a')
    s.set_linewidth(1.2)
ax1.set_xticks([]); ax1.set_yticks([])

# Badge inside card
ax1.text(0.08, 0.92, "[ TECHNIQUE 01 ]", fontfamily='monospace', fontsize=10, weight='bold', color='#2547eb', transform=ax1.transAxes)
ax1.text(0.08, 0.84, "ROBUST SCALER", fontfamily='sans-serif', fontsize=20, weight='black', color='#0f172a', transform=ax1.transAxes)
ax1.text(0.08, 0.77, "Chuẩn hóa theo Median & IQR", fontfamily='monospace', fontsize=11, color='#64748b', transform=ax1.transAxes)

# Stat highlight box
rect1 = patches.Rectangle((0.08, 0.58), 0.84, 0.15, facecolor='#f1f5f9', edgecolor='#0f172a', linewidth=0.8, transform=ax1.transAxes)
ax1.add_patch(rect1)
ax1.text(0.12, 0.67, "BÁC BỎ MIN-MAX", fontfamily='sans-serif', fontsize=11, weight='bold', color='#dc2626', transform=ax1.transAxes)
ax1.text(0.12, 0.61, "Ngoại lai 25$ nén bẹp dữ liệu thường", fontfamily='sans-serif', fontsize=10, color='#334155', transform=ax1.transAxes)

# Key points
bullet1 = (
    "• Công thức: x' = (x - Median) / IQR\n\n"
    "• Miễn nhiễm Outliers: Đơn mua đột biến\n"
    "  (Salad $25 - $30) không kéo lệch thang đo.\n\n"
    "• Giữ nguyên phân phối: Bảo tồn nguyên vẹn\n"
    "  khoảng cách giá trị cốt lõi của quán."
)
ax1.text(0.08, 0.50, bullet1, fontfamily='sans-serif', fontsize=11.5, color='#1e293b', va='top', transform=ax1.transAxes, linespacing=1.4)


# ==============================================================================
# CARD 2: CATEGORICAL ENCODING & THE DUMMY TRAP
# ==============================================================================
ax2 = fig.add_axes([0.36, 0.10, 0.28, 0.66])
ax2.set_facecolor('white')
for s in ax2.spines.values():
    s.set_color('#0f172a')
    s.set_linewidth(1.2)
ax2.set_xticks([]); ax2.set_yticks([])

ax2.text(0.08, 0.92, "[ TECHNIQUE 02 ]", fontfamily='monospace', fontsize=10, weight='bold', color='#2547eb', transform=ax2.transAxes)
ax2.text(0.08, 0.84, "ONE-HOT ENCODING", fontfamily='sans-serif', fontsize=20, weight='black', color='#0f172a', transform=ax2.transAxes)
ax2.text(0.08, 0.77, "Mã hóa biến phân loại an toàn", fontfamily='monospace', fontsize=11, color='#64748b', transform=ax2.transAxes)

# Stat highlight box
rect2 = patches.Rectangle((0.08, 0.58), 0.84, 0.15, facecolor='#f1f5f9', edgecolor='#0f172a', linewidth=0.8, transform=ax2.transAxes)
ax2.add_patch(rect2)
ax2.text(0.12, 0.67, "THE DUMMY TRAP", fontfamily='sans-serif', fontsize=11, weight='bold', color='#d97706', transform=ax2.transAxes)
ax2.text(0.12, 0.61, "Bẫy đa cộng tuyến hoàn hảo giữa các cột", fontfamily='sans-serif', fontsize=10, color='#334155', transform=ax2.transAxes)

bullet2 = (
    "• Thả rơi cột cơ sở: drop_first = True\n\n"
    "• Logic thực tế: Không Tại quán (0)\n"
    "  thì chắc chắn là Mang về (1).\n\n"
    "• Triệt tiêu nhiễu: Loại bỏ cột dư thừa,\n"
    "  giúp mô hình tính toán ổn định, không lỗi."
)
ax2.text(0.08, 0.50, bullet2, fontfamily='sans-serif', fontsize=11.5, color='#1e293b', va='top', transform=ax2.transAxes, linespacing=1.4)


# ==============================================================================
# CARD 3: DATA LEAKAGE (Golden Principle)
# ==============================================================================
ax3 = fig.add_axes([0.67, 0.10, 0.28, 0.66])
ax3.set_facecolor('#0f172a') # Black/Dark Navy accent card for stark brutalist contrast!
for s in ax3.spines.values():
    s.set_color('#0f172a')
ax3.set_xticks([]); ax3.set_yticks([])

ax3.text(0.08, 0.92, "[ GOLDEN RULE ]", fontfamily='monospace', fontsize=10, weight='bold', color='#38bdf8', transform=ax3.transAxes)
ax3.text(0.08, 0.84, "ANTI-DATA LEAKAGE", fontfamily='sans-serif', fontsize=20, weight='black', color='white', transform=ax3.transAxes)
ax3.text(0.08, 0.77, "Chống rò rỉ dữ liệu tương lai", fontfamily='monospace', fontsize=11, color='#94a3b8', transform=ax3.transAxes)

# Highlight box inside dark card
rect3 = patches.Rectangle((0.08, 0.58), 0.84, 0.15, facecolor='#1e293b', edgecolor='#38bdf8', linewidth=0.8, transform=ax3.transAxes)
ax3.add_patch(rect3)
ax3.text(0.12, 0.67, "CHIA TRƯỚC - FIT SAU", fontfamily='sans-serif', fontsize=11, weight='bold', color='#38bdf8', transform=ax3.transAxes)
ax3.text(0.12, 0.61, "Time-based Train/Test Split độc lập", fontfamily='sans-serif', fontsize=10, color='#cbd5e1', transform=ax3.transAxes)

bullet3 = (
    "• Rủi ro chết người: Fit Scaler trên toàn\n"
    "  bộ dữ liệu làm lộ trước tương lai cho mô hình.\n\n"
    "• Chuẩn thực thi:\n"
    "  1. Chia Train/Test theo trục thời gian.\n"
    "  2. CHỈ fit scaler trên tập Train (quá khứ).\n"
    "  3. Áp dụng cùng tham số sang tập Test.\n\n"
    "• Kết quả: Đánh giá thực chất, không ảo!"
)
ax3.text(0.08, 0.50, bullet3, fontfamily='sans-serif', fontsize=11.5, color='#e2e8f0', va='top', transform=ax3.transAxes, linespacing=1.35)

plt.savefig('/tmp/sample_type_shift_slide5.png', bbox_inches='tight', dpi=150)
plt.close()
print("Sample Type Shift Slide 5 created successfully.")
