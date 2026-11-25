import os
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

from src.business_report import render_business_report
from src.missing_details import render_missing_details
from src.outlier_noise_details import render_outlier_noise_details
from src.eda import (
    plot_category_performance,
    plot_discount_vs_profit,
    plot_sales_heatmap,
    plot_sales_over_time,
    plot_top_subcategories,
)
from src.model import forecast_future_sales, prepare_time_series_data, train_forecast_model
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

    # Khai báo đầy đủ 5 Tab tích hợp toàn diện phân tích và chất lượng dữ liệu
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analytics & Insights", 
        "🤖 Sales Forecasting (Time Series)", 
        "📑 Business Report",
        "🔍 Missing Data Details",
        "🎯 Outlier & Noise Details"
    ])

    # =========================================================================
    # TAB 1: EDA (An) & STATISTICAL TESTING (Đức)
    # =========================================================================
    with tab1:
        # --- Khối xuất file dữ liệu đã làm sạch ---
        col_export_info, col_export_btn = st.columns([3, 1])
        with col_export_info:
            st.markdown("#### 📥 Xuất File Dữ Liệu Đã Làm Sạch")
            st.caption(f"Tập dữ liệu đã chuẩn hóa qua 6 bước tiền xử lý & làm sạch (Hiện có **{len(df_filtered):,}** dòng sẵn sàng trích xuất).")
        with col_export_btn:
            csv_bytes = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Xuất file đã sạch (CSV)",
                data=csv_bytes,
                file_name="cleaned_cafe_sales.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
                help="Tải xuống tập dữ liệu cleaned_cafe_sales.csv đã xử lý khuyết thiếu, nhiễu và ngoại lai"
            )
        st.markdown("---")

        total_sales = df_filtered['sales'].sum() if 'sales' in df_filtered.columns else 0.0
        total_orders = df_filtered['order_id'].nunique() if 'order_id' in df_filtered.columns else len(df_filtered)
        total_qty = df_filtered['quantity'].sum() if 'quantity' in df_filtered.columns else 0.0
        avg_order_value = (total_sales / total_orders) if total_orders > 0 else 0.0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng Doanh Thu", f"${total_sales:,.2f}")
        col2.metric("Tổng Số Đơn Hàng", f"{total_orders:,}")
        col3.metric("Tổng Sản Phẩm Đã Bán", f"{int(total_qty):,}")
        col4.metric("Giá Trị Đơn TB (AOV)", f"${avg_order_value:,.2f}")
        
        st.markdown("---")
        
        st.subheader("📈 Phân tích Xu hướng & Hiệu suất Doanh thu")
        st.plotly_chart(plot_sales_over_time(df_filtered), width="stretch")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(plot_category_performance(df_filtered), width="stretch")
        with col_right:
            if 'payment_method' in df_filtered.columns:
                import plotly.express as px
                fig_pm = px.pie(
                    df_filtered, 
                    names='payment_method', 
                    values='sales', 
                    title='💳 Tỷ trọng Doanh thu theo Phương thức Thanh toán',
                    hole=0.4
                )
                fig_pm.update_layout(template='plotly_white')
                st.plotly_chart(fig_pm, width="stretch")
            else:
                st.plotly_chart(plot_discount_vs_profit(df_filtered), width="stretch")

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            fig_sub = plot_top_subcategories(df_filtered)
            st.plotly_chart(fig_sub, width="stretch")
        with col_eda2:
            fig_heat = plot_sales_heatmap(df_filtered)
            st.plotly_chart(fig_heat, width="stretch")

        # Phân hệ Kiểm định thống kê (chỉ kích hoạt nếu có trường chiết khấu & lợi nhuận)
        if 'discount' in df_filtered.columns and 'profit' in df_filtered.columns:
            st.markdown("---")
            render_stats(df_filtered, discount_col='discount', profit_col='profit')

        st.markdown("---")
        st.markdown("### 📋 Dữ liệu mẫu (Top 10 dòng)")
        st.dataframe(df_filtered.head(10), width="stretch")

    # =========================================================================
    # TAB 2: MACHINE LEARNING SALES FORECASTING (Trí)
    # =========================================================================
    with tab2:
        st.subheader("💡 Dự Báo Doanh Số Theo Ngày")
        st.write("Holt-Winters học chu kỳ **7 ngày** từ dữ liệu quán cà phê hiện có. Dự báo bắt đầu ngay sau ngày cuối cùng trong dữ liệu lịch sử.")
        
        col_in1, col_in2 = st.columns([1, 2])
        with col_in1:
            st.markdown("### 🎛️ Bảng Điều Khiển Kịch Bản")
            cat_input = st.selectbox(
                "Chọn Ngành hàng (Category):", 
                options=["All"] + (categories if categories else ["Technology"])
            )
            days_input = st.slider(
                "Số ngày dự báo:",
                min_value=7,
                max_value=28,
                value=14,
                step=7,
            )
            run_sim = st.button("🚀 Chạy Dự Báo", width="stretch", type="primary")

        with col_in2:
            try:
                ts_hist = prepare_time_series_data(df, category=cat_input)
                import plotly.graph_objects as go

                if ts_hist.empty:
                    st.warning("Không có giao dịch hợp lệ cho danh mục đã chọn.")
                elif run_sim:
                    model, metrics = train_forecast_model(df, category=cat_input, horizon=days_input)
                    forecast_df = forecast_future_sales(model, steps=days_input)

                    st.markdown("### 🎯 Kết Quả Đánh Giá & Dự Báo")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    total_forecast = forecast_df['predicted_sales'].sum()
                    avg_forecast = forecast_df['predicted_sales'].mean()
                    m_col1.metric("Tổng Doanh Số Dự Báo", f"${total_forecast:,.2f}")
                    m_col2.metric("Trung bình / Ngày", f"${avg_forecast:,.2f}")
                    m_col3.metric("MAE / Ngày", f"${metrics['mae']:,.2f}")
                    st.caption(
                        f"Đánh giá theo thời gian trên {metrics['folds']} khoảng liên tiếp, "
                        f"mỗi khoảng {days_input} ngày. Holt-Winters: MAE ${metrics['mae']:,.2f}/ngày, "
                        f"WAPE {metrics['wape']:.1f}%; cách dự báo cùng thứ tuần trước: "
                        f"MAE ${metrics['baseline_mae']:,.2f}/ngày, WAPE {metrics['baseline_wape']:.1f}%."
                    )
                    st.caption("Ngày không có giao dịch ghi nhận cho danh mục được tính là 0. "
                               "Dự báo nối tiếp dữ liệu năm 2023, không phải doanh thu hiện tại.")
                    st.markdown("---")
                    st.markdown(f"#### 📈 Biểu đồ Lịch sử và Dự báo Doanh số ({cat_input})")
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=ts_hist.tail(90).index,
                        y=ts_hist['sales'].tail(90),
                        mode='lines',
                        name='Thực tế (90 ngày cuối)',
                        line=dict(color='royalblue', width=2)
                    ))
                    last_hist_date = ts_hist.index[-1]
                    last_hist_val = ts_hist['sales'].iloc[-1]
                    forecast_dates = [last_hist_date] + list(forecast_df['forecast_date'])
                    forecast_vals = [last_hist_val] + list(forecast_df['predicted_sales'])
                    fig_ts.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_vals,
                        mode='lines+markers',
                        name='Dự báo theo ngày',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    fig_ts.update_layout(
                        template='plotly_white',
                        xaxis_title="Ngày",
                        yaxis_title="Doanh số / ngày ($)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_ts, width="stretch")
                    st.markdown("#### ⚖️ Bảng Dữ Liệu Tương Lai")
                    st.dataframe(
                        forecast_df.rename(columns={
                            'forecast_date': 'Ngày',
                            'predicted_sales': 'Doanh số dự báo ($)'
                        }).style.format({'Doanh số dự báo ($)': "${:,.2f}"}),
                        width="stretch"
                    )
                    if metrics['mae'] >= metrics['baseline_mae']:
                        st.warning("Holt-Winters chưa tốt hơn cách dự báo cùng thứ tuần trước ở các khoảng kiểm tra này.")
                else:
                    st.info(f"Dữ liệu lịch sử: {len(ts_hist)} ngày, đến {ts_hist.index.max():%d/%m/%Y}. Nhấn Chạy Dự Báo để xem kết quả.")
            except ValueError as exc:
                st.warning(str(exc))

    # =========================================================================
    # TAB 3: BUSINESS REPORT (Mạnh)
    # =========================================================================
    with tab3:
        render_business_report(df_filtered)

    # =========================================================================
    # TAB 4: MISSING DATA DETAILS
    # =========================================================================
    with tab4:
        render_missing_details()

    # =========================================================================
    # TAB 5: OUTLIER & NOISE DETAILS
    # =========================================================================
    with tab5:
        render_outlier_noise_details()

else:
    st.warning("⚠️ Chưa tìm thấy dữ liệu trong Database. Vui lòng kiểm tra file `data/data.db` hoặc chạy `src/pipeline.py`!")
