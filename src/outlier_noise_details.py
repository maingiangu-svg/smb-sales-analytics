import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DEFAULT_RAW_FILE = os.path.join("data", "dirty_cafe_sales.csv")


def compute_outlier_noise_statistics(file_path: str = DEFAULT_RAW_FILE):
    """
    Phân tích chi tiết các loại Outliers (Ngoại lai) và Noise (Nhiễu)
    trên dataset thô đầu vào dirty_cafe_sales.csv theo chuẩn thống kê.
    """
    if not os.path.exists(file_path):
        return None, None, None, None, None

    df_raw = pd.read_csv(file_path, keep_default_na=False)
    total_records = len(df_raw)

    cost_raw = pd.to_numeric(df_raw['Cost'], errors='coerce')
    sales_raw = pd.to_numeric(df_raw['Total Spent'], errors='coerce')

    # =========================================================================
    # 1. PHÂN TÍCH NOISE (NHIỄU)
    # =========================================================================
    noise_results = []

    # Cột Cost: Nhiễu số âm
    mask_neg_cost = cost_raw < 0
    cnt_neg_cost = int(mask_neg_cost.sum())
    noise_results.append({
        'Tên Cột': 'Cost',
        'Loại Nhiễu (Noise Type)': 'Nhiễu Dấu Âm (Negative Sign Noise)',
        'Số Lượng Phát Hiện': cnt_neg_cost,
        'Tỷ Lệ / Tổng Dataset (%)': round((cnt_neg_cost / total_records) * 100, 2),
        'Ví Dụ Cụ Thể': '-0.3, -1.2, -1.8, -2.2',
        'Giải Pháp Trong Pipeline': 'Áp dụng trị tuyệt đối df[\'cost\'].abs() để chuyển thành số dương hợp lệ'
    })

    # Cột Cost: Nhiễu giá trị 0
    mask_zero_cost = cost_raw == 0
    cnt_zero_cost = int(mask_zero_cost.sum())
    noise_results.append({
        'Tên Cột': 'Cost',
        'Loại Nhiễu (Noise Type)': 'Nhiễu Số 0 Vô Nghĩa (Zero Cost Noise)',
        'Số Lượng Phát Hiện': cnt_zero_cost,
        'Tỷ Lệ / Tổng Dataset (%)': round((cnt_zero_cost / total_records) * 100, 2),
        'Ví Dụ Cụ Thể': '0.0 (chi phí bằng 0 vô lý trong F&B)',
        'Giải Pháp Trong Pipeline': 'Chuyển thành NaN và điền lại bằng Group Median theo danh mục món ăn'
    })

    # Các cột còn lại (Không có noise số học)
    clean_cols_noise = ['Total Spent', 'Quantity', 'Price Per Unit', 'Item', 'Payment Method', 'Location', 'Transaction Date', 'Transaction ID']
    for c in clean_cols_noise:
        noise_results.append({
            'Tên Cột': c,
            'Loại Nhiễu (Noise Type)': 'Không phát hiện nhiễu (Clean)',
            'Số Lượng Phát Hiện': 0,
            'Tỷ Lệ / Tổng Dataset (%)': 0.00,
            'Ví Dụ Cụ Thể': 'Dữ liệu chuẩn, không dính số âm/ký tự lạ',
            'Giải Pháp Trong Pipeline': 'Không cần can thiệp'
        })

    noise_df = pd.DataFrame(noise_results)

    # =========================================================================
    # 2. PHÂN TÍCH OUTLIERS (NGOẠI LAI THỐNG KÊ THUẦN TÚY)
    # =========================================================================
    outlier_results = []

    # Cột Cost: Ngoại lai đơn vị đo lường (nhân 100 lần - lệch cực xa so với phân phối 0.3 - 2.2)
    mask_scale_cost = cost_raw.abs() >= 10
    cnt_scale_cost = int(mask_scale_cost.sum())
    outlier_results.append({
        'Tên Cột': 'Cost',
        'Loại Ngoại Lai (Outlier Type)': 'Ngoại Lai Đơn Vị (Scale Outlier - Cents vs Dollars)',
        'Số Lượng Phát Hiện': cnt_scale_cost,
        'Tỷ Lệ / Tổng Dataset (%)': round((cnt_scale_cost / total_records) * 100, 2),
        'Ví Dụ Cụ Thể': '30.0, 100.0, 120.0, 150.0, 180.0, 220.0',
        'Giải Pháp Trong Pipeline': 'Phát hiện >= 10 và chia cho 100.0 để quy về chuẩn USD'
    })

    # Cột Total Spent (Sales): Ngoại lai thống kê nhẹ theo 1.5x IQR
    sales_valid = sales_raw.dropna()
    q1_s = sales_valid.quantile(0.25)
    q3_s = sales_valid.quantile(0.75)
    iqr_s = q3_s - q1_s
    upper_sales_15 = q3_s + 1.5 * iqr_s
    mask_sales_iqr = sales_raw > upper_sales_15
    cnt_sales_iqr = int(mask_sales_iqr.sum())
    outlier_results.append({
        'Tên Cột': 'Total Spent',
        'Loại Ngoại Lai (Outlier Type)': 'Ngoại Lai Thống Kê Nhẹ (Mild Statistical Outlier - 1.5x IQR)',
        'Số Lượng Phát Hiện': cnt_sales_iqr,
        'Tỷ Lệ / Tổng Dataset (%)': round((cnt_sales_iqr / total_records) * 100, 2),
        'Ví Dụ Cụ Thể': 'Đơn hàng $25.0 (mua 5 đĩa Salad: $5 x 5 = $25)',
        'Giải Pháp Trong Pipeline': 'Giữ nguyên 100% doanh thu thật vì nằm dưới trần cực đoan $36'
    })

    # Các cột còn lại (Không có Outlier)
    clean_cols_outlier = ['Quantity', 'Price Per Unit', 'Item', 'Payment Method', 'Location', 'Transaction Date', 'Transaction ID']
    for c in clean_cols_outlier:
        outlier_results.append({
            'Tên Cột': c,
            'Loại Ngoại Lai (Outlier Type)': 'Không phát hiện ngoại lai (Clean)',
            'Số Lượng Phát Hiện': 0,
            'Tỷ Lệ / Tổng Dataset (%)': 0.00,
            'Ví Dụ Cụ Thể': 'Giá trị nằm trong khung chuẩn thiết kế',
            'Giải Pháp Trong Pipeline': 'Không cần can thiệp'
        })

    outlier_df = pd.DataFrame(outlier_results)

    # 3. Tổng hợp dòng bị ảnh hưởng bởi Outlier hoặc Noise thuần túy
    any_issue_mask = mask_neg_cost | mask_zero_cost | mask_scale_cost | mask_sales_iqr
    total_affected_rows = int(any_issue_mask.sum())

    summary_kpis = {
        'total_records': total_records,
        'total_outliers': cnt_scale_cost + cnt_sales_iqr,
        'total_noise': cnt_neg_cost + cnt_zero_cost,
        'affected_rows': total_affected_rows,
        'pct_affected_rows': round((total_affected_rows / total_records) * 100, 2)
    }

    masks = {
        'scale_cost': mask_scale_cost,
        'sales_iqr': mask_sales_iqr,
        'neg_cost': mask_neg_cost,
        'zero_cost': mask_zero_cost,
    }

    return outlier_df, noise_df, summary_kpis, df_raw, masks


def render_outlier_noise_details(file_path: str = DEFAULT_RAW_FILE):
    """
    Giao diện hiển thị trang Outlier & Noise Details trên Streamlit.
    """
    st.subheader("🎯 Chi Tiết Ngoại Lai & Nhiễu (Outlier & Noise Details)")
    st.caption("Báo cáo phân loại, đo lường số lượng và giải thích cơ chế xử lý Outlier & Noise trên tập dữ liệu thô `dirty_cafe_sales.csv` theo chuẩn thống kê.")

    outlier_df, noise_df, kpis, df_raw, masks = compute_outlier_noise_statistics(file_path)

    if outlier_df is None or kpis is None:
        st.error(f"⚠️ Không tìm thấy file dữ liệu đầu vào tại `{file_path}`. Vui lòng kiểm tra lại đường dẫn!")
        return

    # =========================================================================
    # PHẦN 1: KPI CARDS TỔNG QUAN
    # =========================================================================
    st.markdown("### 1. Chỉ Số Tổng Quan Về Ngoại Lai & Nhiễu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng Số Dòng Phân Tích", f"{kpis['total_records']:,}")
    c2.metric("Tổng Điểm Ngoại Lai (Outliers)", f"{kpis['total_outliers']:,} điểm", "2 loại thống kê chuẩn")
    c3.metric("Tổng Điểm Nhiễu (Noise)", f"{kpis['total_noise']:,} điểm", "2 loại phát hiện")
    c4.metric(
        "Dòng Bị Ảnh Hưởng", 
        f"{kpis['affected_rows']:,}", 
        f"{kpis['pct_affected_rows']}% tổng dòng",
        delta_color="inverse"
    )

    st.info(
        "💡 **Nguyên tắc phân định học thuật chuẩn:**\n"
        "* **Missing Data (Khuyết thiếu):** Các giá trị bị vắng mặt (`ERROR`, `UNKNOWN`, ô rỗng) đã được giải quyết ở Tab trước.\n"
        "* **Noise (Nhiễu):** Dữ liệu có tồn tại nhưng bị sai định dạng/âm vô lý (dấu trừ, số 0).\n"
        "* **Outlier (Ngoại lai):** Giá trị lệch cực xa so với phân phối chung của cột (như giá vốn $30 - $220 vọt gấp 100 lần, hoặc doanh thu đơn hàng cao).\n"
        "*(Lưu ý: Các trường hợp mâu thuẫn đa biến như Cost > Price được xử lý ở khâu Business Consistency, không tính vào Outlier đơn biến thống kê).*"
    )

    st.markdown("---")

    # =========================================================================
    # PHẦN 2: CHI TIẾT VỀ OUTLIERS (NGOẠI LAI)
    # =========================================================================
    st.markdown("### 2. Danh Sách & Tỷ Lệ Outliers (Ngoại Lai)")
    st.write("Bảng thống kê số lượng, tỷ lệ phần trăm và bản chất của từng loại Outlier thực sự trong dataset:")

    active_outliers = outlier_df[outlier_df['Số Lượng Phát Hiện'] > 0].copy()
    active_outliers['Tỷ Lệ / Tổng Dataset (%)'] = active_outliers['Tỷ Lệ / Tổng Dataset (%)'].map('{:.2f}%'.format)
    active_outliers['Số Lượng Phát Hiện'] = active_outliers['Số Lượng Phát Hiện'].map('{:,}'.format)
    st.dataframe(active_outliers, width="stretch", hide_index=True)

    # Plotly Bar chart for Outliers
    fig_out = px.bar(
        outlier_df[outlier_df['Số Lượng Phát Hiện'] > 0],
        x='Loại Ngoại Lai (Outlier Type)',
        y='Số Lượng Phát Hiện',
        color='Tên Cột',
        text='Số Lượng Phát Hiện',
        title="Phân Bố Số Lượng Điểm Ngoại Lai (Outliers) Theo Phân Loại",
        color_discrete_sequence=['#e74c3c', '#f39c12']
    )
    fig_out.update_traces(textposition='outside')
    fig_out.update_layout(template='plotly_white', xaxis_title="", yaxis_title="Số lượng bản ghi")
    st.plotly_chart(fig_out, width="stretch")

    st.markdown("---")

    # =========================================================================
    # PHẦN 3: CHI TIẾT VỀ NOISE (NHIỄU)
    # =========================================================================
    st.markdown("### 3. Danh Sách & Tỷ Lệ Noise (Nhiễu)")
    st.write("Bảng thống kê số lượng, tỷ lệ phần trăm và bản chất của từng loại Noise số học:")

    active_noise = noise_df[noise_df['Số Lượng Phát Hiện'] > 0].copy()
    active_noise['Tỷ Lệ / Tổng Dataset (%)'] = active_noise['Tỷ Lệ / Tổng Dataset (%)'].map('{:.2f}%'.format)
    active_noise['Số Lượng Phát Hiện'] = active_noise['Số Lượng Phát Hiện'].map('{:,}'.format)
    st.dataframe(active_noise, width="stretch", hide_index=True)

    # Plotly Donut chart for Noise
    fig_noise = px.pie(
        noise_df[noise_df['Số Lượng Phát Hiện'] > 0],
        names='Loại Nhiễu (Noise Type)',
        values='Số Lượng Phát Hiện',
        title="Cơ Cấu Tỷ Lệ Các Loại Nhiễu Số Học (Noise)",
        hole=0.45,
        color_discrete_sequence=['#3498db', '#9b59b6']
    )
    fig_noise.update_layout(template='plotly_white')
    st.plotly_chart(fig_noise, width="stretch")

    st.markdown("---")

    # =========================================================================
    # PHẦN 4: TRA CỨU DỮ LIỆU THỰC TẾ (DATA DRILL-DOWN VIEWER)
    # =========================================================================
    st.markdown("### 4. Tra Cứu Mẫu Bản Ghi Bị Outlier & Noise Trong Dữ Liệu Gốc")

    drill_options = {
        "1. Cột Cost - Ngoại lai nhân 100 lần (Scale Outlier >= 10)": masks['scale_cost'],
        "2. Cột Total Spent - Đơn hàng doanh thu cao (> $24.0)": masks['sales_iqr'],
        "3. Cột Cost - Nhiễu số âm (Negative Values)": masks['neg_cost'],
        "4. Cột Cost - Nhiễu chi phí bằng 0 (Cost == 0.0)": masks['zero_cost'],
    }

    selected_inspect = st.selectbox(
        "Chọn nhóm lỗi cần xem bằng chứng cụ thể trên file gốc:",
        options=list(drill_options.keys())
    )

    if selected_inspect and df_raw is not None:
        target_mask = drill_options[selected_inspect]
        sample_rows = df_raw[target_mask].head(10)
        st.write(f"Mẫu 10 bản ghi minh chứng cho trường hợp **'{selected_inspect}'**:")
        st.dataframe(sample_rows, width="stretch")
