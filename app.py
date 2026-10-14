import os
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

from src.business_report import render_business_report
from src.eda import (
    plot_category_performance,
    plot_discount_vs_profit,
    plot_sales_heatmap,
    plot_sales_over_time,
    plot_top_subcategories,
)
from src.model import load_or_train_model, forecast_future_sales, prepare_time_series_data
from src.stats import render_streamlit as render_stats

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

    # Khai báo đầy đủ 3 Tab tích hợp công việc của cả 4 thành viên
    tab1, tab2, tab3 = st.tabs([
        "📊 Analytics & Insights", 
        "🤖 Sales Forecasting (Time Series)", 
        "📑 Business Report"
    ])

    # =========================================================================
    # TAB 1: EDA (An) & STATISTICAL TESTING (Đức)
    # =========================================================================
    with tab1:
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
        
        st.subheader("📈 Phân tích Doanh thu & Lợi nhuận")
        st.plotly_chart(plot_sales_over_time(df_filtered), width="stretch")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(plot_category_performance(df_filtered), width="stretch")
        with col_right:
            st.plotly_chart(plot_discount_vs_profit(df_filtered), width="stretch")

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            fig_sub = plot_top_subcategories(df_filtered)
            st.plotly_chart(fig_sub, width="stretch")
        with col_eda2:
            fig_heat = plot_sales_heatmap(df_filtered)
            st.plotly_chart(fig_heat, width="stretch")

        st.markdown("---")
        
        # Phân hệ Kiểm định thống kê của Đức
        render_stats(df_filtered, discount_col='discount', profit_col='profit')

        st.markdown("---")
        st.markdown("### 📋 Dữ liệu mẫu (Top 10 dòng)")
        st.dataframe(df_filtered.head(10), width="stretch")

    # =========================================================================
    # TAB 2: MACHINE LEARNING SALES FORECASTING (Trí)
    # =========================================================================
    with tab2:
        st.subheader("💡 Dự Báo Doanh Số Tương Lai (Time Series Forecasting)")
        st.write("Hệ thống sử dụng mô hình học máy chuỗi thời gian **Holt-Winters Exponential Smoothing** để dự báo xu hướng doanh số trong các tháng tiếp theo dựa trên dữ liệu lịch sử.")
        
        col_in1, col_in2 = st.columns([1, 2])
        with col_in1:
            st.markdown("### 🎛️ Bảng Điều Khiển Kịch Bản")
            cat_input = st.selectbox(
                "Chọn Ngành hàng (Category):", 
                options=["All"] + (categories if categories else ["Technology"])
            )
            months_input = st.slider(
                "Số tháng dự báo tương lai:", 
                min_value=3, 
                max_value=12, 
                value=6, 
                step=1
            )
            
            run_sim = st.button("🚀 Chạy Dự Báo", width="stretch", type="primary")

        with col_in2:
            try:
                # 1. Hiển thị dữ liệu lịch sử
                ts_hist = prepare_time_series_data(df, freq='ME', category=cat_input)
                
                # 2. Load model and predict
                import plotly.graph_objects as go
                
                if not ts_hist.empty and len(ts_hist) >= 6:
                    # Train model dynamically for the specific category if needed, 
                    # or just load the base model if 'All'
                    from src.model import train_forecast_model
                    model, metrics = train_forecast_model(df, category=cat_input)
                    
                    forecast_df = forecast_future_sales(model, steps=months_input)
                    
                    st.markdown("### 🎯 Kết Quả Đánh Giá & Dự Báo")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    
                    total_forecast = forecast_df['predicted_sales'].sum()
                    avg_forecast = forecast_df['predicted_sales'].mean()
                    
                    m_col1.metric("Tổng Doanh Số Dự Báo", f"${total_forecast:,.2f}")
                    m_col2.metric("Trung bình / Tháng", f"${avg_forecast:,.2f}")
                    m_col3.metric("Sai Số (MAE)", f"${metrics['mae']:,.2f}")
                    
                    st.markdown("---")
                    st.markdown(f"#### 📈 Biểu đồ Lịch sử và Dự báo Doanh số ({cat_input})")
                    
                    fig_ts = go.Figure()
                    
                    # Dữ liệu lịch sử
                    fig_ts.add_trace(go.Scatter(
                        x=ts_hist.index,
                        y=ts_hist['sales'],
                        mode='lines+markers',
                        name='Thực tế (Historical)',
                        line=dict(color='royalblue', width=2)
                    ))
                    
                    # Dữ liệu dự báo
                    # Nối điểm cuối cùng của lịch sử với điểm đầu của dự báo để không đứt đoạn
                    last_hist_date = ts_hist.index[-1]
                    last_hist_val = ts_hist['sales'].iloc[-1]
                    
                    forecast_dates = [last_hist_date] + list(forecast_df['forecast_date'])
                    forecast_vals = [last_hist_val] + list(forecast_df['predicted_sales'])
                    
                    fig_ts.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_vals,
                        mode='lines+markers',
                        name='Dự báo (Forecast)',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig_ts.update_layout(
                        template='plotly_white',
                        xaxis_title="Thời gian",
                        yaxis_title="Doanh số ($)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_ts, width="stretch")
                    
                    st.markdown("#### ⚖️ Bảng Dữ Liệu Tương Lai")
                    st.dataframe(
                        forecast_df.rename(columns={
                            'forecast_date': 'Tháng',
                            'predicted_sales': 'Doanh số dự báo ($)'
                        }).style.format({'Doanh số dự báo ($)': "${:,.2f}"}),
                        width="stretch"
                    )
                    
                else:
                    st.warning(f"⚠️ Dữ liệu lịch sử cho danh mục {cat_input} quá ít (dưới 6 tháng), không đủ để chạy thuật toán Time Series.")
                
            except Exception as e:
                st.error(f"⚠️ Lỗi khi chạy mô hình dự báo Time Series: {e}")

    # =========================================================================
    # TAB 3: BUSINESS REPORT (Mạnh)
    # =========================================================================
    with tab3:
        render_business_report(df_filtered)

else:
    st.warning("⚠️ Chưa tìm thấy dữ liệu trong Database. Vui lòng kiểm tra file `data/data.db` hoặc chạy `src/pipeline.py`!")