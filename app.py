import streamlit as st
import pandas as pd
import sqlite3
import os

from src.eda import (
    plot_sales_over_time,
    plot_category_performance,
    plot_discount_vs_profit,
    plot_top_subcategories,
    plot_sales_heatmap
)
from src.stats import render_streamlit as render_statistical_tests

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
        
        # Convert order_date sang datetime ngay khi load DB
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Lỗi kết nối CSDL: {e}")
        return None

st.title("📊 SMB Sales Analytics & Pricing Optimization Platform")
st.markdown("---")

df = load_data_from_db()

if df is not None and not df.empty:
    # Sidebar: Bộ lọc dữ liệu
    st.sidebar.header("🔍 Bộ Lọc Dữ Liệu")
    categories = df['category'].dropna().unique().tolist() if 'category' in df.columns else []
    selected_category = st.sidebar.multiselect(
        "Chọn Danh Mục:", 
        options=categories,
        default=categories
    )

    if selected_category and 'category' in df.columns:
        df_filtered = df[df['category'].isin(selected_category)]
    else:
        df_filtered = df.copy()

    # Phân chia Tabs
    tab1, tab2 = st.tabs(["📊 Analytics & Insights", "🤖 Pricing Simulator (ML)"])

    with tab1:
        # Hàng KPI tổng quan
        total_sales = df_filtered['sales'].sum() if 'sales' in df_filtered.columns else 0.0
        total_profit = df_filtered['profit'].sum() if 'profit' in df_filtered.columns else 0.0
        total_orders = df_filtered['order_id'].nunique() if 'order_id' in df_filtered.columns else len(df_filtered)
        avg_discount = (df_filtered['discount'].mean() * 100) if 'discount' in df_filtered.columns else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng Doanh Thu", f"${total_sales:,.2f}")
        col2.metric("Tổng Lợi Nhuận", f"${total_profit:,.2f}")
        col3.metric("Tổng Số Đơn Hàng", f"{total_orders:,}")
        col4.metric("Chiết Khấu TB", f"{avg_discount:.1f}%")
        
        st.markdown("---")
        
        # Biểu đồ xu hướng thời gian
        st.subheader("📈 Phân tích Doanh thu & Lợi nhuận")
        st.plotly_chart(plot_sales_over_time(df_filtered), width="stretch")
        
        # Cặp biểu đồ phân bổ danh mục & tương quan chiết khấu
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(plot_category_performance(df_filtered), width="stretch")
        with col_right:
            st.plotly_chart(plot_discount_vs_profit(df_filtered), width="stretch")

        # Cặp biểu đồ nâng cao mới (Top Sub-category & Heatmap)
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.plotly_chart(plot_top_subcategories(df_filtered), width="stretch")
        with col_eda2:
            st.plotly_chart(plot_sales_heatmap(df_filtered), width="stretch")
                    st.markdown("---")
        st.subheader("🧪 Kiểm định thống kê: Chiết khấu và Lợi nhuận")
        render_statistical_tests(df_filtered, discount_col="discount", profit_col="profit")

        st.markdown("---")
        st.markdown("### 📋 Dữ liệu mẫu (Top 10 dòng)")
        st.dataframe(df_filtered.head(10), width="stretch")

    with tab2:
        st.subheader("💡 Gợi Ý Mức Chiết Khấu Tối Ưu Lợi Nhuận")
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            cat_input = st.selectbox(
                "Ngành hàng:", 
                options=categories if categories else ["Technology"]
            )
            sales_input = st.number_input("Doanh số dự kiến ($):", min_value=0.0, value=500.0, step=10.0)
        with col_in2:
            st.info("Module ML đang được tích hợp từ branch `feature/ml-pricing-model`...")
            if st.button("Dự báo Discount tối ưu"):
                st.warning("Đang chờ hàm predict_optimal_discount() từ src/model.py")

else:
    st.warning("⚠️ Chưa tìm thấy dữ liệu trong Database. Vui lòng kiểm tra file `data/data.db` hoặc chạy `src/pipeline.py`!")
