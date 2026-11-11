
import pandas as pd
import streamlit as st
import plotly.express as px

def render_business_report(df: pd.DataFrame):
    """
    Hiển thị Báo cáo Kinh doanh SMB tổng hợp từ dữ liệu bán hàng.
    """
    if df is None or df.empty:
        st.warning("⚠️ Không có dữ liệu để tạo Business Report.")
        return

    st.subheader("📑 Báo Cáo Phân Tích Hiệu Quả Kinh Doanh (Business Report)")
    st.caption("Báo cáo tóm tắt chỉ số tài chính, cấu trúc biên lợi nhuận và kiến nghị chiến lược định giá.")

    # 1. Các chỉ số tài chính tổng hợp
    total_sales = df['sales'].sum() if 'sales' in df.columns else 0.0
    total_profit = df['profit'].sum() if 'profit' in df.columns else 0.0
    total_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0.0
    overall_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0.0

    st.markdown("### 1. Chỉ số Tài chính Cốt lõi (Executive Summary)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Doanh Thu Thuần", f"${total_sales:,.2f}")
    c2.metric("Lợi Nhuận Gộp", f"${total_profit:,.2f}")
    c3.metric("Biên Lợi Nhuận TB", f"{overall_margin:.2f}%")
    c4.metric("Giá Trị Đơn Trung Bình (AOV)", f"${avg_order_value:,.2f}")

    st.markdown("---")

    # 2. Phân tích chi tiết theo danh mục sản phẩm
    st.markdown("### 2. Hiệu Suất Theo Danh Mục Mặt Hàng")
    if 'category' in df.columns and 'sales' in df.columns and 'profit' in df.columns:
        cat_summary = df.groupby('category').agg(
            Doanh_Thu=('sales', 'sum'),
            Loi_Nhuan=('profit', 'sum'),
            So_Don=('order_id', 'nunique' if 'order_id' in df.columns else 'count'),
            Chiet_Khau_TB=('discount', lambda x: x.mean() * 100 if 'discount' in df.columns else 0.0)
        ).reset_index()

        cat_summary['Bien_Loi_Nhuan_%'] = (cat_summary['Loi_Nhuan'] / cat_summary['Doanh_Thu'] * 100).round(2)
        cat_summary['Doanh_Thu'] = cat_summary['Doanh_Thu'].map('${:,.2f}'.format)
        cat_summary['Loi_Nhuan'] = cat_summary['Loi_LoiNhuan' if 'Loi_LoiNhuan' in cat_summary else 'Loi_Nhuan'].map('${:,.2f}'.format)
        cat_summary['Chiet_Khau_TB'] = cat_summary['Chiet_Khau_TB'].map('{:.1f}%'.format)

        st.dataframe(cat_summary, width="stretch", hide_index=True)

    st.markdown("---")

    # 3. Đánh giá rủi ro chiết khấu & Khuyến nghị hành động
    st.markdown("### 3. Đánh Giá Tác Động Chiết Khấu & Đề Xuất Chiến Lược")
    
    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        st.error("🚨 **Rủi ro phát hiện từ dữ liệu:**")
        st.markdown("""
        * **Hiện tượng 'Bào mòn lợi nhuận':** Các đơn hàng áp dụng chiết khấu sâu (>20%) có xu hướng làm sụt giảm mạnh biên lợi nhuận ròng.
        * **Lãng phí khuyến mãi:** Một số mặt hàng có nhu cầu tự nhiên cao nhưng vẫn bị gán chiết khấu mặc định, làm mất biên lãi không cần thiết.
        """)

    with col_rec2:
        st.success("💡 **Kiến nghị hành động cho SMB:**")
        st.markdown("""
        * **Áp trần chiết khấu (Discount Ceiling):** Giới hạn chiết khấu tối đa ở mức tối ưu được đề xuất bởi mô hình Machine Learning ở Tab 2.
        * **Tập trung biên lợi nhuận:** Ưu tiên đẩy mạnh các nhóm mặt hàng có tỷ suất lợi nhuận cao thay vì chỉ chạy theo tăng trưởng doanh thu thô.
        """)