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

st.title("📊 SMB Sales Analytics & Pricing Optimization Platform")
st.markdown("---")

df = load_data_from_db()

if df is not None and not df.empty:
    st.success(f"✅ Đã tải thành công **{len(df):,}** bản ghi từ Database!")
    
    total_sales = df['sales'].sum() if 'sales' in df.columns else 0.0
    total_profit = df['profit'].sum() if 'profit' in df.columns else 0.0
    total_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
    avg_discount = (df['discount'].mean() * 100) if 'discount' in df.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng Doanh Thu", f"${total_sales:,.2f}")
    col2.metric("Tổng Lợi Nhuận", f"${total_profit:,.2f}")
    col3.metric("Tổng Số Đơn Hàng", f"{total_orders:,}")
    col4.metric("Chiết Khấu TB", f"{avg_discount:.1f}%")
    
    st.markdown("---")
    
    # 📈 THÊM PHẦN RENDER BIỂU ĐỒ Ở ĐÂY
    st.subheader("📈 Phân tích Doanh thu & Lợi nhuận")
    st.plotly_chart(plot_sales_over_time(df), use_container_width=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(plot_category_performance(df), use_container_width=True)
    with col_right:
        st.plotly_chart(plot_discount_vs_profit(df), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Dữ liệu mẫu (Top 10 dòng)")
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.warning("⚠️ Chưa tìm thấy dữ liệu trong Database. Vui lòng chạy Pipeline hoặc Upload file dữ liệu thô!")
# Cập nhật đoạn lọc và chia Tab trong app.py
st.sidebar.header("🔍 Bộ Lọc Dữ Liệu")
selected_category = st.sidebar.multiselect(
    "Chọn Danh Mục:", 
    options=df['category'].unique() if 'category' in df.columns else [],
    default=df['category'].unique() if 'category' in df.columns else []
)

if selected_category and 'category' in df.columns:
    df_filtered = df[df['category'].isin(selected_category)]
else:
    df_filtered = df.copy()

tab1, tab2 = st.tabs(["📊 Analytics & Insights", "🤖 Pricing Simulator (ML)"])

with tab1:
    # Đưa các KPI và Chart hiện tại vào đây (dùng df_filtered)
    ...

with tab2:
    st.subheader("💡 Gợi Ý Mức Chiết Khấu Tối Ưu Lợi Nhuận")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        cat_input = st.selectbox("Ngành hàng:", df['category'].unique() if 'category' in df.columns else ["Technology"])
        sales_input = st.number_input("Doanh số dự kiến ($):", min_value=0.0, value=500.0)
    with col_in2:
        st.info("Module ML đang được tích hợp từ branch `feature/ml-pricing-model`...")
        if st.button("Dự báo Discount tối ưu"):
            st.warning("Đang chờ hàm predict_optimal_discount() từ src/model.py")
@st.cache_data
def load_data_from_db():
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM sales_data", conn)
        conn.close()
        
        # Convert order_date sang datetime ngay khi load DB
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Lỗi kết nối CSDL: {e}")
        return None