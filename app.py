import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px

from src.eda import (
    plot_sales_over_time,
    plot_category_performance,
    plot_discount_vs_profit,
    plot_top_subcategories,
    plot_sales_heatmap
)
from src.model import load_or_train_model, optimize_discount

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
        st.markdown("### 📋 Dữ liệu mẫu (Top 10 dòng)")
        st.dataframe(df_filtered.head(10), width="stretch")

    with tab2:
        st.subheader("💡 Gợi Ý Mức Chiết Khấu Tối Ưu Lợi Nhuận (Machine Learning)")
        st.write("Hệ thống sử dụng mô hình Machine Learning **RandomForestRegressor** được huấn luyện trên dữ liệu doanh số để mô phỏng và tìm ra mức Chiết khấu (Discount) tối đa hóa Lợi nhuận.")
        
        col_in1, col_in2 = st.columns([1, 2])
        with col_in1:
            st.markdown("### 🎛️ Bảng Điều Khiển Kịch Bản")
            cat_input = st.selectbox(
                "Chọn Ngành hàng (Category):", 
                options=categories if categories else ["Technology"]
            )
            sales_input = st.number_input(
                "Doanh số dự kiến ($):", 
                min_value=10.0, 
                value=500.0, 
                step=10.0
            )
            
            run_sim = st.button("🚀 Phân Tích & Gợi Ý Chiết Khấu Tối Ưu", width="stretch", type="primary")

        with col_in2:
            try:
                model = load_or_train_model()
                opt_result = optimize_discount(model, cat_input, sales_input)
                
                sim_df = opt_result['simulation_df']
                opt_discount_pct = opt_result['optimal_discount_pct']
                max_profit = opt_result['max_profit']
                profit_margin = opt_result['profit_margin_pct']
                
                st.markdown("### 🎯 Kết Quả Mô Phỏng & Gợi Ý")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Chiết Khấu Tối Ưu", f"{opt_discount_pct:.1f}%")
                m_col2.metric("Lợi Nhuận Dự Báo Tối Đa", f"${max_profit:,.2f}")
                m_col3.metric("Tỷ Tỉ Lệ Lợi Nhuận (Margin)", f"{profit_margin:.1f}%")
                
                st.markdown("---")
                st.markdown("#### 📈 Đường Cong Lợi Nhuận Theo Mức Chiết Khấu (Profit vs Discount Curve)")
                
                # Tạo biểu đồ đường bằng Plotly
                sim_df_plot = sim_df.copy()
                sim_df_plot['discount_pct'] = sim_df_plot['discount'] * 100
                
                fig_curve = px.line(
                    sim_df_plot,
                    x='discount_pct',
                    y='predicted_profit',
                    title=f"Mô phỏng Lợi nhuận ngành [{cat_input}] ở mức Doanh số ${sales_input:,.2f}",
                    labels={'discount_pct': 'Tỷ lệ Chiết khấu (%)', 'predicted_profit': 'Lợi nhuận dự báo ($)'}
                )
                
                # Highlight điểm cực đại
                fig_curve.add_scatter(
                    x=[opt_discount_pct],
                    y=[max_profit],
                    mode='markers+text',
                    marker=dict(size=14, color='red', symbol='star'),
                    text=[f"Đỉnh Lợi Nhuận: {opt_discount_pct}% (${max_profit:,.2f})"],
                    textposition="top center",
                    name="Mức tối ưu"
                )
                fig_curve.update_layout(template='plotly_white')
                st.plotly_chart(fig_curve, width="stretch")
                
                # Bảng so sánh 3 kịch bản
                st.markdown("#### ⚖️ Bảng So Sánh Kịch Bản Chiết Khấu")
                baseline_row = sim_df.loc[sim_df['discount'] == 0.0].iloc[0]
                std_row = sim_df.loc[(sim_df['discount'] - 0.15).abs().idxmin()]
                opt_row = sim_df.loc[sim_df['discount'] == opt_result['optimal_discount']].iloc[0]
                
                scenario_df = pd.DataFrame([
                    {
                        "Kịch Bản": "Không Chiết Khấu (0%)",
                        "Mức Chiết Khấu (%)": "0.0%",
                        "Lợi Nhuận Dự Báo ($)": f"${baseline_row['predicted_profit']:,.2f}",
                        "Biên Lợi Nhuận (%)": f"{baseline_row['margin_pct']:.1f}%",
                        "Chênh Lệch So Với 0%": "$0.00"
                    },
                    {
                        "Kịch Bản": "Mặc Định Ngành (15%)",
                        "Mức Chiết Khấu (%)": "15.0%",
                        "Lợi Nhuận Dự Báo ($)": f"${std_row['predicted_profit']:,.2f}",
                        "Biên Lợi Nhuận (%)": f"{std_row['margin_pct']:.1f}%",
                        "Chênh Lệch So Với 0%": f"${std_row['predicted_profit'] - baseline_row['predicted_profit']:+,.2f}"
                    },
                    {
                        "Kịch Bản": "⭐ Gợi Ý Tối Ưu ML",
                        "Mức Chiết Khấu (%)": f"{opt_discount_pct:.1f}%",
                        "Lợi Nhuận Dự Báo ($)": f"${max_profit:,.2f}",
                        "Biên Lợi Nhuận (%)": f"{profit_margin:.1f}%",
                        "Chênh Lệch So Với 0%": f"${max_profit - baseline_row['predicted_profit']:+,.2f}"
                    }
                ])
                st.dataframe(scenario_df, width="stretch", hide_index=True)
                
            except Exception as e:
                st.error(f"⚠️ Lỗi khi mô phỏng mô hình ML: {e}")

else:
    st.warning("⚠️ Chưa tìm thấy dữ liệu trong Database. Vui lòng kiểm tra file `data/data.db` hoặc chạy `src/pipeline.py`!")