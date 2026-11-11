import pandas as pd
import streamlit as st
import plotly.express as px

def render_business_report(df: pd.DataFrame):
    """
    Hiển thị Báo cáo Hiệu quả Kinh doanh & Doanh số tổng hợp từ dữ liệu bán hàng.
    """
    if df is None or df.empty:
        st.warning("⚠️ Không có dữ liệu để tạo Business Report.")
        return

    st.subheader("📑 Báo Cáo Phân Tích Hiệu Quả Kinh Doanh & Bán Hàng (Business Report)")
    st.caption("Báo cáo tóm tắt chỉ số doanh thu, cơ cấu mặt hàng và kiến nghị chiến lược vận hành.")

    # 1. Các chỉ số doanh thu cốt lõi
    total_sales = df['sales'].sum() if 'sales' in df.columns else 0.0
    total_orders = df['order_id'].nunique() if 'order_id' in df.columns else len(df)
    total_qty = df['quantity'].sum() if 'quantity' in df.columns else 0.0
    avg_order_value = (total_sales / total_orders) if total_orders > 0 else 0.0

    st.markdown("### 1. Chỉ số Kinh doanh Cốt lõi (Executive Summary)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Doanh Thu Thuần", f"${total_sales:,.2f}")
    c2.metric("Tổng Số Đơn Hàng", f"{total_orders:,}")
    c3.metric("Tổng Sản Phẩm Đã Bán", f"{int(total_qty):,}")
    c4.metric("Giá Trị Đơn Trung Bình (AOV)", f"${avg_order_value:,.2f}")

    st.markdown("---")

    # 2. Phân tích chi tiết theo danh mục sản phẩm
    st.markdown("### 2. Hiệu Suất Bán Hàng Theo Danh Mục Mặt Hàng")
    if 'category' in df.columns and 'sales' in df.columns:
        cat_summary = df.groupby('category').agg(
            Doanh_Thu=('sales', 'sum'),
            So_Don=('order_id', 'nunique' if 'order_id' in df.columns else 'count'),
            So_Luong=('quantity', 'sum' if 'quantity' in df.columns else 'count'),
            Don_Gia_TB=('price_per_unit', 'mean' if 'price_per_unit' in df.columns else lambda x: 0.0)
        ).reset_index()

        cat_summary['Ty_Trong_Doanh_Thu_%'] = (cat_summary['Doanh_Thu'] / total_sales * 100).round(2) if total_sales > 0 else 0.0
        cat_summary['Doanh_Thu'] = cat_summary['Doanh_Thu'].map('${:,.2f}'.format)
        cat_summary['Don_Gia_TB'] = cat_summary['Don_Gia_TB'].map('${:,.2f}'.format)
        cat_summary['So_Don'] = cat_summary['So_Don'].map('{:,}'.format)
        cat_summary['So_Luong'] = cat_summary['So_Luong'].astype(int).map('{:,}'.format)
        cat_summary['Ty_Trong_Doanh_Thu_%'] = cat_summary['Ty_Trong_Doanh_Thu_%'].map('{:.1f}%'.format)

        st.dataframe(cat_summary, width="stretch", hide_index=True)

    st.markdown("---")

    # 3. Đánh giá vận hành & Khuyến nghị hành động
    st.markdown("### 3. Đánh Giá Xu Hướng Doanh Số & Đề Xuất Chiến Lược Vận Hành")
    
    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        st.info("💡 **Ghi nhận từ mô hình chuỗi thời gian & hành vi:**")
        st.markdown("""
        * **Tính mùa vụ & chu kỳ theo tuần:** Doanh số có sự dao động rõ rệt giữa các ngày trong tuần và các tháng cao điểm.
        * **Đóng góp doanh thu chủ lực:** Nhóm đồ uống (Coffee, Smoothie, Juice) và bánh đóng góp phần lớn tỷ trọng doanh thu toàn quán.
        """)

    with col_rec2:
        st.success("🎯 **Kiến nghị hành động cho Quán:**")
        st.markdown("""
        * **Dự báo nhu cầu nguyên vật liệu:** Dựa vào kết quả dự báo doanh thu ở Tab 2 để chủ động chuẩn bị nguyên phụ liệu trước các tháng cao điểm.
        * **Tối ưu hóa giá trị đơn hàng (AOV):** Áp dụng chiến lược combo (ví dụ: Cà phê + Bánh ngọt) để nâng cao giá trị trung bình trên mỗi lượt giao dịch.
        """)
