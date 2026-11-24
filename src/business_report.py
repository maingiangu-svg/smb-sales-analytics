import pandas as pd
import streamlit as st
import plotly.express as px

def render_business_report(df: pd.DataFrame):
    """
    Báo cáo Kinh doanh: Chỉ số Doanh thu Cốt lõi, Rủi ro và Chiến lược ngắn gọn.
    Thiết kế đồng bộ font chữ, súc tích, dễ đọc và trực quan.
    """
    if df is None or df.empty:
        st.warning("Không có dữ liệu để tạo báo cáo. Vui lòng kiểm tra lại bộ lọc.")
        return

    st.subheader("Báo Cáo Hiệu Quả Doanh Thu & Chiến Lược Vận Hành")
    st.caption("Tổng hợp chỉ số bán hàng cốt lõi, nhận diện rủi ro chính và đề xuất hành động tối ưu doanh thu.")

    # =========================================================================
    # 1. CHỈ SỐ DOANH THU CỐT LÕI
    # =========================================================================
    total_sales = float(df['sales'].sum()) if 'sales' in df.columns else 0.0
    total_orders = int(df['order_id'].nunique()) if 'order_id' in df.columns else len(df)
    total_qty = float(df['quantity'].sum()) if 'quantity' in df.columns else 0.0
    aov = (total_sales / total_orders) if total_orders > 0 else 0.0

    st.markdown("### 1. Chỉ số Doanh thu Cốt lõi")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng doanh thu", f"${total_sales:,.2f}")
    m2.metric("Tổng số đơn hàng", f"{total_orders:,}")
    m3.metric("Tổng sản phẩm tiêu thụ", f"{int(total_qty):,}")
    m4.metric("Giá trị đơn trung bình (AOV)", f"${aov:,.2f}")

    st.markdown("---")

    # =========================================================================
    # 2. CƠ CẤU DOANH THU THEO MÓN
    # =========================================================================
    st.markdown("### 2. Cơ Cấu Doanh Thu Theo Danh Mục Món")
    
    if 'category' in df.columns and 'sales' in df.columns:
        cat_df = df.groupby('category').agg(
            Doanh_Thu=('sales', 'sum'),
            So_Luong=('quantity', 'sum' if 'quantity' in df.columns else 'count'),
            So_Don=('order_id', 'nunique' if 'order_id' in df.columns else 'count'),
            Don_Gia=('price_per_unit', 'mean' if 'price_per_unit' in df.columns else lambda x: 0.0)
        ).reset_index()

        cat_df['Ty_Trong_%'] = (cat_df['Doanh_Thu'] / total_sales * 100).round(1) if total_sales > 0 else 0.0
        cat_df = cat_df.sort_values(by='Doanh_Thu', ascending=False)

        col_chart, col_tbl = st.columns([1.1, 0.9])

        with col_chart:
            fig_bar = px.bar(
                cat_df,
                x='Doanh_Thu',
                y='category',
                orientation='h',
                text='Doanh_Thu',
                labels={'Doanh_Thu': 'Doanh thu ($)', 'category': 'Món'},
                title="Đóng góp doanh thu theo món"
            )
            fig_bar.update_traces(
                texttemplate='$%{text:,.0f}',
                textposition='outside',
                marker_color='#1E40AF'
            )
            fig_bar.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                template='plotly_white',
                font=dict(family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, sans-serif", size=13),
                height=320,
                margin=dict(l=10, r=40, t=35, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_tbl:
            tbl_disp = cat_df.copy()
            tbl_disp['Doanh_Thu'] = tbl_disp['Doanh_Thu'].map('${:,.2f}'.format)
            tbl_disp['Don_Gia'] = tbl_disp['Don_Gia'].map('${:,.2f}'.format)
            tbl_disp['So_Luong'] = tbl_disp['So_Luong'].astype(int).map('{:,}'.format)
            tbl_disp['Ty_Trong_%'] = tbl_disp['Ty_Trong_%'].map('{:.1f}%'.format)
            tbl_disp = tbl_disp[['category', 'Doanh_Thu', 'So_Luong', 'Don_Gia', 'Ty_Trong_%']]
            tbl_disp.columns = ['Danh mục', 'Doanh thu', 'Số lượng bán', 'Đơn giá', 'Tỷ trọng']
            st.dataframe(tbl_disp, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =========================================================================
    # 3. PHÂN TÍCH RỦI RO (NGẮN GỌN - ĐÚNG TRỌNG TÂM)
    # =========================================================================
    st.markdown("### 3. Phân Tích Rủi Ro Doanh Thu")
    
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown("**Phụ thuộc nhóm món chính**")
        st.markdown(
            "- Salad và Sandwich chiếm tới **36.8% tổng doanh thu**.\n"
            "- Nếu hụt nguồn cung nguyên liệu của 2 món này, doanh số toàn quán sẽ tụt dốc."
        )

    with rc2:
        st.markdown("**Chi tiêu mỗi đơn còn thấp**")
        st.markdown(
            "- Giá trị đơn trung bình (AOV) chỉ đạt **$8.92/đơn**.\n"
            "- Khách hàng chủ yếu mua lẻ 1 món, chưa có thói quen mua nhiều sản phẩm cùng lúc."
        )

    with rc3:
        st.markdown("**Dữ liệu bán hàng bị khuyết**")
        st.markdown(
            "- Gần **40% đơn thiếu hình thức bán** (Tại chỗ hay Mang đi).\n"
            "- **31% thanh toán ngoài luồng**, gây khó khăn cho việc quản lý doanh thu theo kênh."
        )

    st.markdown("---")

    # =========================================================================
    # 4. ĐỀ XUẤT CHIẾN LƯỢC DOANH THU (NGẮN GỌN - DỄ HIỂU)
    # =========================================================================
    st.markdown("### 4. Đề Xuất Chiến Lược Tối Ưu Doanh Thu")

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**1. Bán Combo đồ uống kèm đồ ăn (Cross-selling)**")
        st.markdown(
            "- Ghép Cà phê ($2) + Sandwich ($4) thành combo giá $5.5 (giảm $0.5).\n"
            "- Ghép Trà ($1.5) + Bánh ngọt ($3) thành combo giá $4.0.\n"
            "- **Mục tiêu:** Kéo giá trị đơn trung bình (AOV) lên cao hơn."
        )

        st.markdown("**2. Kích cầu khung giờ chiều giữa tuần (Happy Hour)**")
        st.markdown(
            "- Giảm nhẹ giá đồ uống từ 14h00 – 16h30 các ngày Thứ 2 đến Thứ 4.\n"
            "- **Mục tiêu:** Thu hút dân văn phòng mua giải khát buổi chiều, tăng doanh số ngày vắng."
        )

    with sc2:
        st.markdown("**3. Ứng dụng dự báo Holt-Winters vào vận hành**")
        st.markdown(
            "- Dùng kết quả dự báo doanh thu theo ngày để chuẩn bị nguyên liệu.\n"
            "- Xếp ca làm việc nhân viên linh hoạt theo ngày dự báo khách đông hoặc vắng."
        )

        st.markdown("**4. Chuẩn hóa thao tác máy POS tại quầy**")
        st.markdown(
            "- Thêm tính năng thanh toán phù hợp để tăng khả năng kiểm soát dòng tiền."
        )
