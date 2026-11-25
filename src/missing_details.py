import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DEFAULT_RAW_FILE = os.path.join("data", "dirty_cafe_sales.csv")


def compute_missing_statistics(file_path: str = DEFAULT_RAW_FILE):
    """
    Phân tích chi tiết các loại missing values (ERROR, UNKNOWN, Rỗng)
    và tính toán số dòng bị ảnh hưởng trên dataset thô đầu vào.
    """
    if not os.path.exists(file_path):
        return None, None, None

    # Đọc với keep_default_na=False để giữ nguyên chuỗi rỗng ''
    df_raw = pd.read_csv(file_path, keep_default_na=False)
    total_records = len(df_raw)

    results = []
    is_missing_matrix = pd.DataFrame(index=df_raw.index)

    # Từ điển ánh xạ nhận định cơ chế khuyết thiếu theo lý thuyết thống kê
    mechanism_map = {
        'Transaction ID': 'Valid (Không thiếu)',
        'Item': 'MAR (UNKNOWN/Rỗng) + MCAR (ERROR)',
        'Quantity': 'MAR (UNKNOWN/Rỗng) + MCAR (ERROR)',
        'Price Per Unit': 'MAR (UNKNOWN/Rỗng) + MCAR (ERROR)',
        'Total Spent': 'MAR (UNKNOWN/Rỗng) + MCAR (ERROR)',
        'Payment Method': 'MNAR (UNKNOWN/Rỗng) + MCAR (ERROR)',
        'Location': 'MAR (UNKNOWN/Rỗng) + MCAR (ERROR)',
        'Transaction Date': 'MCAR (Lỗi đồng hồ/NTP hệ thống)',
        'Cost': 'MAR (UNKNOWN/Rỗng) + MCAR (ERROR)'
    }

    for col in df_raw.columns:
        series_str = df_raw[col].astype(str).str.strip()

        # 1. Đếm ERROR (không phân biệt hoa thường)
        cnt_error = int((series_str.str.upper() == 'ERROR').sum())

        # 2. Đếm UNKNOWN (không phân biệt hoa thường)
        cnt_unknown = int((series_str.str.upper() == 'UNKNOWN').sum())

        # 3. Đếm Rỗng / NaN / None / Null
        empty_mask = series_str.isin(['', 'nan', 'NAN', 'NaN', 'None', 'NONE', 'null', 'NULL'])
        cnt_empty = int(empty_mask.sum())

        # Mặt nạ tổng missing của cột này
        col_missing_mask = (
            series_str.str.upper().isin(['ERROR', 'UNKNOWN', 'NAN', 'NONE', 'NULL']) | 
            (series_str == '')
        )
        is_missing_matrix[col] = col_missing_mask

        total_missing = int(col_missing_mask.sum())
        missing_rate = (total_missing / total_records) * 100 if total_records > 0 else 0.0

        results.append({
            'Tên Cột (Column)': col,
            'Số lượng ERROR': cnt_error,
            'Số lượng UNKNOWN': cnt_unknown,
            'Số ô Rỗng / NaN': cnt_empty,
            'Tổng Missing Values': total_missing,
            'Tỷ lệ Missing (%)': round(missing_rate, 2),
            'Cơ Chế Phân Loại': mechanism_map.get(col, 'Unknown')
        })

    summary_df = pd.DataFrame(results)

    # 2. Thống kê số dòng có ít nhất 1 ô missing
    rows_with_missing_mask = is_missing_matrix.any(axis=1)
    rows_with_at_least_one = int(rows_with_missing_mask.sum())
    pct_rows_with_missing = (rows_with_at_least_one / total_records) * 100 if total_records > 0 else 0.0
    complete_rows = total_records - rows_with_at_least_one
    pct_complete_rows = 100.0 - pct_rows_with_missing

    row_stats = {
        'total_records': total_records,
        'rows_with_missing': rows_with_at_least_one,
        'pct_rows_with_missing': round(pct_rows_with_missing, 2),
        'complete_rows': complete_rows,
        'pct_complete_rows': round(pct_complete_rows, 2),
        'total_missing_cells': int(summary_df['Tổng Missing Values'].sum())
    }

    return summary_df, row_stats, df_raw


def render_missing_details(file_path: str = DEFAULT_RAW_FILE):
    """
    Giao diện hiển thị trang Missing Data Details trên Streamlit.
    """
    st.subheader("🔍 Chi Tiết Dữ Liệu Khuyết Thiếu (Missing Data Details)")
    st.caption("Báo cáo kiểm toán chất lượng dữ liệu và phân tích cơ chế khuyết thiếu (MCAR, MAR, MNAR) dựa trên tập dữ liệu thô `dirty_cafe_sales.csv`.")

    summary_df, row_stats, df_raw = compute_missing_statistics(file_path)

    if summary_df is None or row_stats is None:
        st.error(f"⚠️ Không tìm thấy file dữ liệu đầu vào tại `{file_path}`. Vui lòng kiểm tra lại đường dẫn!")
        return

    # =========================================================================
    # PHẦN 1: KPI CARDS TỔNG QUAN
    # =========================================================================
    st.markdown("### 1. Chỉ Số Khuyết Thiếu Toàn Diện (Overview Metrics)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tổng Số Dòng Dữ Liệu", f"{row_stats['total_records']:,}")
    k2.metric(
        "Dòng Bị Khuyết (≥ 1 ô)", 
        f"{row_stats['rows_with_missing']:,}", 
        f"{row_stats['pct_rows_with_missing']}% tổng dòng",
        delta_color="inverse"
    )
    k3.metric(
        "Dòng Hoàn Chỉnh (0 ô thiếu)", 
        f"{row_stats['complete_rows']:,}", 
        f"{row_stats['pct_complete_rows']}% tổng dòng"
    )
    k4.metric("Tổng Số Ô Bị Khuyết", f"{row_stats['total_missing_cells']:,} ô")

    # Cảnh báo nghiệp vụ sâu sắc
    st.info(
        f"💡 **Thông điệp cốt lõi từ dữ liệu:** Có tới **{row_stats['rows_with_missing']:,} trên tổng số {row_stats['total_records']:,} dòng ({row_stats['pct_rows_with_missing']}%)** "
        f"chứa ít nhất 1 ô bị khuyết (ERROR, UNKNOWN hoặc rỗng). Nếu sử dụng phương pháp loại bỏ dòng ngây thơ (`dropna`), "
        f"hệ thống sẽ **đánh mất gần 75% dữ liệu**, chỉ còn lại {row_stats['pct_complete_rows']}% bản ghi! "
        f"Do đó, việc phân loại chính xác các cơ chế **MCAR, MAR, MNAR** để khôi phục chéo là điều kiện tiên quyết."
    )

    st.markdown("---")

    # =========================================================================
    # PHẦN 2: BẢNG LIỆT KÊ CHI TIẾT TỪNG CỘT
    # =========================================================================
    st.markdown("### 2. Thống Kê Chi Tiết Số Lượng & Tỷ Lệ Khuyết Thiếu Theo Từng Cột")
    st.write("Bảng phân rã chi tiết 3 loại missing: **ERROR** (lỗi kỹ thuật/hệ thống), **UNKNOWN** (chưa xác định), và **Rỗng / NaN**.")

    # Format hiển thị
    styled_df = summary_df.copy()
    styled_df['Tỷ lệ Missing (%)'] = styled_df['Tỷ lệ Missing (%)'].map('{:.2f}%'.format)
    styled_df['Số lượng ERROR'] = styled_df['Số lượng ERROR'].map('{:,}'.format)
    styled_df['Số lượng UNKNOWN'] = styled_df['Số lượng UNKNOWN'].map('{:,}'.format)
    styled_df['Số ô Rỗng / NaN'] = styled_df['Số ô Rỗng / NaN'].map('{:,}'.format)
    styled_df['Tổng Missing Values'] = styled_df['Tổng Missing Values'].map('{:,}'.format)

    st.dataframe(styled_df, width="stretch", hide_index=True)

    st.markdown("---")

    # =========================================================================
    # PHẦN 3: TRỰC QUAN HÓA CƠ CẤU VÀ TỶ LỆ MISSING (PLOTLY)
    # =========================================================================
    st.markdown("### 3. Trực Quan Hóa Tỷ Lệ & Thành Phần Missing Values")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Bar chart: Tỷ lệ missing theo từng cột
        plot_df = summary_df.sort_values(by='Tỷ lệ Missing (%)', ascending=True)
        fig_bar = px.bar(
            plot_df,
            x='Tỷ lệ Missing (%)',
            y='Tên Cột (Column)',
            orientation='h',
            text='Tỷ lệ Missing (%)',
            title="Tỷ Lệ Thiếu Dữ Liệu Theo Cột (%)",
            color='Tỷ lệ Missing (%)',
            color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_bar.update_layout(template='plotly_white', xaxis_title="Tỷ lệ (%)", yaxis_title="")
        st.plotly_chart(fig_bar, width="stretch")

    with col_chart2:
        # Stacked bar chart: Phân rã cấu trúc ERROR vs UNKNOWN vs Rỗng
        fig_stacked = go.Figure()
        fig_stacked.add_trace(go.Bar(
            y=summary_df['Tên Cột (Column)'],
            x=summary_df['Số lượng ERROR'],
            name='ERROR (MCAR)',
            orientation='h',
            marker=dict(color='#e74c3c')
        ))
        fig_stacked.add_trace(go.Bar(
            y=summary_df['Tên Cột (Column)'],
            x=summary_df['Số lượng UNKNOWN'],
            name='UNKNOWN (MAR/MNAR)',
            orientation='h',
            marker=dict(color='#f39c12')
        ))
        fig_stacked.add_trace(go.Bar(
            y=summary_df['Tên Cột (Column)'],
            x=summary_df['Số ô Rỗng / NaN'],
            name='Rỗng / NaN (MAR/MCAR)',
            orientation='h',
            marker=dict(color='#3498db')
        ))
        fig_stacked.update_layout(
            barmode='stack',
            title="Phân Rã Cơ Cấu Lỗi (ERROR vs UNKNOWN vs Rỗng)",
            template='plotly_white',
            xaxis_title="Số lượng ô",
            yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_stacked, width="stretch")

    st.markdown("---")

    # =========================================================================
    # PHẦN 4: TRA CỨU DỮ LIỆU THỰC TẾ (SAMPLE DRILL-DOWN)
    # =========================================================================
    st.markdown("### 4. Tra Cứu Dòng Dữ Liệu Bị Khuyết Thô (Data Inspection)")
    selected_col = st.selectbox(
        "Chọn cột để kiểm tra mẫu các dòng bị khuyết (ERROR / UNKNOWN / Rỗng):",
        options=[c for c in summary_df['Tên Cột (Column)'] if c != 'Transaction ID']
    )

    if selected_col and df_raw is not None:
        series_str = df_raw[selected_col].astype(str).str.strip()
        mask_error = series_str.str.upper() == 'ERROR'
        mask_unknown = series_str.str.upper() == 'UNKNOWN'
        mask_empty = series_str.isin(['', 'nan', 'NAN', 'NaN', 'None', 'NONE', 'null', 'NULL'])
        
        sample_rows = df_raw[mask_error | mask_unknown | mask_empty].head(10)
        st.write(f"Mẫu 10 giao dịch chứa dữ liệu lỗi/khuyết tại cột `{selected_col}`:")
        st.dataframe(sample_rows, width="stretch")
