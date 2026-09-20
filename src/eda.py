import plotly.express as px
import pandas as pd

def plot_sales_over_time(df):
    """Biểu đồ xu hướng Doanh thu & Lợi nhuận hàng tháng"""
    cols = [c for c in ['sales', 'profit'] if c in df.columns]
    if not cols or 'order_date' not in df.columns:
        return px.line(title="⚠️ Thiếu dữ liệu thời gian")
    
    # 1. Tạo bản sao để tránh chỉnh sửa df gốc
    df_temp = df.copy()
    
    # 2. Ép kiểu datetime cho order_date (xử lý triệt để lỗi TypeError)
    df_temp['order_date'] = pd.to_datetime(df_temp['order_date'], errors='coerce')
    df_temp = df_temp.dropna(subset=['order_date'])
    
    if df_temp.empty:
        return px.line(title="⚠️ Không có dữ liệu thời gian hợp lệ")

    # 3. Gom nhóm theo tháng (Month End)
    df_monthly = (
        df_temp.set_index('order_date')
        .resample('ME')[cols]
        .sum()
        .reset_index()
    )
    
    fig = px.line(
        df_monthly, 
        x='order_date', 
        y=cols if len(cols) > 1 else cols[0],
        title='📈 Xu hướng Doanh thu & Lợi nhuận hàng tháng',
        labels={'value': 'USD ($)', 'order_date': 'Thời gian', 'variable': 'Chỉ số'}
    )
    fig.update_layout(template='plotly_white')
    return fig
def plot_category_performance(df):
    """Biểu đồ Doanh thu & Lợi nhuận theo Danh mục sản phẩm (Category)"""
    cols = [c for c in ['sales', 'profit'] if c in df.columns]
    cat_col = 'category' if 'category' in df.columns else None
    
    if not cat_col or not cols:
        return px.bar(title="⚠️ Không tìm thấy dữ liệu Danh mục")
        
    df_cat = df.groupby(cat_col)[cols].sum().reset_index()
    fig = px.bar(
        df_cat, 
        x=cat_col, 
        y=cols if len(cols) > 1 else cols[0],
        barmode='group',
        title='📦 Doanh thu & Lợi nhuận theo Danh mục',
        labels={'value': 'USD ($)', cat_col: 'Danh mục'}
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_discount_vs_profit(df):
    """Scatter Plot: Tác động của Chiết khấu (Discount) tới Lợi nhuận (Profit)"""
    if 'discount' not in df.columns or 'profit' not in df.columns:
        return px.scatter(title="⚠️ Thiếu dữ liệu Discount hoặc Profit để vẽ Scatter Plot")
        
    fig = px.scatter(
        df, 
        x='discount', 
        y='profit', 
        color='category' if 'category' in df.columns else None,
        title='🎯 Mối quan hệ giữa Mức Chiết Khấu & Lợi Nhuận',
        labels={'discount': 'Tỷ lệ chiết khấu (%)', 'profit': 'Lợi nhuận ($)'}
    )
    fig.update_layout(template='plotly_white')
    return fig

#issue #2
# Bổ sung các biểu đồ trực quan hóa dữ liệu nâng cao trong "src/eda.py" bằng Plotly để hiển thị lên Tab EDA.

# Công việc cần làm

# Viết hàm "plot_top_subcategories(df)": Biểu đồ cột ngang Top 10 Sub-category đóng góp lợi nhuận cao nhất.
# Viết hàm "plot_sales_heatmap(df)": Biểu đồ Heatmap doanh số theo Tháng ("month") x Thứ trong tuần ("day_of_week").
# Thêm fallback check dữ liệu đầu vào (tránh crash khi lọc dataframe rỗng).

def plot_top_subcategories(df, top_n=10):
    """
    Biểu đồ cột ngang: Top N Sub-category đóng góp Lợi nhuận cao nhất.
    Chấp nhận cả 'sub_category' và 'subcategory' (tùy cách chuẩn hóa tên cột ở pipeline).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return px.bar(title="⚠️ Không có dữ liệu để hiển thị (dataframe rỗng)")

    subcat_col = next(
        (c for c in ['sub_category', 'subcategory', 'sub-category'] if c in df.columns),
        None
    )

    if not subcat_col or 'profit' not in df.columns:
        return px.bar(title="⚠️ Thiếu dữ liệu Sub-category hoặc Profit")

    df_sub = (
        df.groupby(subcat_col, dropna=True)['profit']
        .sum()
        .reset_index()
        .sort_values('profit', ascending=False)
        .head(top_n)
    )

    if df_sub.empty:
        return px.bar(title="⚠️ Không có dữ liệu Sub-category để hiển thị")

    # Sắp tăng dần để khi vẽ orientation='h', mục cao nhất nằm trên cùng
    df_sub = df_sub.sort_values('profit', ascending=True)

    fig = px.bar(
        df_sub,
        x='profit',
        y=subcat_col,
        orientation='h',
        color='profit',
        color_continuous_scale='RdYlGn',
        title=f'🏆 Top {len(df_sub)} Sub-category theo Lợi nhuận',
        labels={'profit': 'Lợi nhuận ($)', subcat_col: 'Sub-category'}
    )
    fig.update_layout(template='plotly_white', coloraxis_showscale=False)
    return fig

def plot_sales_heatmap(df):
    """
    Heatmap Doanh số (sales) theo Tháng ('month') x Thứ trong tuần ('day_of_week').
    Yêu cầu các cột 'month', 'day_of_week', 'sales' (được sinh ra từ src/pipeline.py).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return px.imshow([[0]], title="⚠️ Không có dữ liệu để hiển thị (dataframe rỗng)")

    required_cols = ['month', 'day_of_week', 'sales']
    if not all(c in df.columns for c in required_cols):
        return px.imshow([[0]], title="⚠️ Thiếu dữ liệu Month/Day_of_week/Sales")

    df_heat = df.copy()

    # Thứ tự chuẩn Thứ 2 -> Chủ nhật (khớp với pd.Series.dt.day_name())
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_heat['day_of_week'] = pd.Categorical(
        df_heat['day_of_week'], categories=day_order, ordered=True
    )
    df_heat = df_heat.dropna(subset=['day_of_week', 'month'])

    if df_heat.empty:
        return px.imshow([[0]], title="⚠️ Không có dữ liệu hợp lệ để vẽ Heatmap")

    pivot = df_heat.pivot_table(
        index='day_of_week',
        columns='month',
        values='sales',
        aggfunc='sum',
        observed=False
    )

    if pivot.empty:
        return px.imshow([[0]], title="⚠️ Không có dữ liệu để hiển thị Heatmap")

    pivot = pivot.reindex(day_order)
    pivot.columns = [f"Tháng {int(m)}" for m in pivot.columns]

    fig = px.imshow(
        pivot,
        labels=dict(x="Tháng", y="Thứ trong tuần", color="Doanh số ($)"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale='YlOrRd',
        aspect='auto',
        title='🔥 Heatmap Doanh số theo Tháng x Thứ trong tuần',
        text_auto='.2s'
    )
    fig.update_layout(template='plotly_white')
    return fig
 