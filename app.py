import streamlit as st
import pandas as pd
import sqlite3
import os
from src.eda import plot_sales_over_time, plot_category_performance, plot_discount_vs_profit
st.set_page_config(
    page_title="SMB Sales Analytics & Pricing", 
    page_icon="📊",
    layout="wide"
)

DB_PATH = os.path.join("data", "data.db")

@st.cache_data
def load_data_from_db():
    """Hàm load data từ SQLite lên Streamlit (có cache để app chạy mượt)"""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM sales_data", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Lỗi kết nối CSDL: {e}")
        return None

# Header
st.title("📊 SMB Sales Analytics & Pricing Optimization Platform")
st.markdown("---")

# Load data từ DB
df = load_data_from_db()

if df is not None and not df.empty:
    st.success(f"✅ Đã tải thành công **{len(df):,}** bản ghi từ Database!")
    
    # Tính toán an toàn chỉ số KPI
    total_sales = df['sales'].sum() if 'sales' in df.columns else 0.0
    total_profit = df['profit'].sum() if 'profit' in df.columns else 0.0
    total_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
    avg_discount = (df['discount'].mean() * 100) if 'discount' in df.columns else 0.0

    # Hiển thị 4 thẻ KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng Doanh Thu", f"${total_sales:,.2f}")
    col2.metric("Tổng Lợi Nhuận", f"${total_profit:,.2f}")
    col3.metric("Tổng Số Đơn Hàng", f"{total_orders:,}")
    col4.metric("Chiết Khấu TB", f"{avg_discount:.1f}%")
    
    st.markdown("---")
    st.markdown("### 📋 Dữ liệu mẫu (Top 10 dòng)")
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.warning("⚠️ Chưa tìm thấy dữ liệu trong Database. Vui lòng chạy Pipeline hoặc Upload file dữ liệu thô!")