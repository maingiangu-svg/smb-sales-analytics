import plotly.express as px
import pandas as pd

def plot_sales_over_time(df):
    """Biểu đồ xu hướng Doanh thu & Lợi nhuận theo ngày."""
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

    # Gom giao dịch theo ngày lịch; giữ ngày không có giao dịch ở mức 0.
    df_daily = (
        df_temp.set_index('order_date')
        .resample('D')[cols]
        .sum()
        .reset_index()
    )
    
    fig = px.line(
        df_daily,
        x='order_date', 
        y=cols if len(cols) > 1 else cols[0],
        title='📈 Xu hướng Doanh thu & Lợi nhuận theo ngày',
        labels={'value': 'USD ($)', 'order_date': 'Ngày', 'variable': 'Chỉ số'}
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
 
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return px.bar(title="⚠️ Không có dữ liệu để hiển thị (dataframe rỗng)")

    subcat_col = next(
        (c for c in ['sub_category', 'subcategory', 'sub-category', 'category'] if c in df.columns),
        None
    )

    metric_col = 'profit' if 'profit' in df.columns else 'sales'
    metric_label = 'Lợi nhuận ($)' if metric_col == 'profit' else 'Doanh thu ($)'

    if not subcat_col or metric_col not in df.columns:
        return px.bar(title="⚠️ Thiếu dữ liệu phân loại hoặc doanh số")

    df_sub = (
        df.groupby(subcat_col, dropna=True)[metric_col]
        .sum()
        .reset_index()
        .sort_values(metric_col, ascending=False)
        .head(top_n)
    )

    if df_sub.empty:
        return px.bar(title="⚠️ Không có dữ liệu để hiển thị")

    # Sắp tăng dần để khi vẽ orientation='h', mục cao nhất nằm trên cùng
    df_sub = df_sub.sort_values(metric_col, ascending=True)

    fig = px.bar(
        df_sub,
        x=metric_col,
        y=subcat_col,
        orientation='h',
        color=metric_col,
        color_continuous_scale='Viridis',
        title=f'🏆 Top {len(df_sub)} Danh mục theo {metric_label}',
        labels={metric_col: metric_label, subcat_col: 'Danh mục'}
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

# ============================================================
# ADVANCED EDA CHARTS
# ============================================================

import plotly.express as px
import pandas as pd


# 1. BOXPLOT + JITTER
# Phân phối doanh thu ngày theo thứ trong tuần
def plot_sales_boxplot_jitter(df):
    """
    Boxplot + Jitter:
    - Trục X: Thứ trong tuần
    - Trục Y: Doanh thu mỗi ngày
    - Mỗi điểm: Một ngày giao dịch
    """

    if df is None or df.empty:
        return px.box(title="⚠️ Không có dữ liệu")

    required = ['order_date', 'sales']
    if not all(col in df.columns for col in required):
        return px.box(title="⚠️ Thiếu order_date hoặc sales")

    temp = df.copy()
    temp['order_date'] = pd.to_datetime(
        temp['order_date'], errors='coerce'
    )
    temp['sales'] = pd.to_numeric(temp['sales'], errors='coerce')

    temp = temp.dropna(subset=['order_date', 'sales'])

    if temp.empty:
        return px.box(title="⚠️ Không có dữ liệu hợp lệ")

    # Tổng doanh thu theo từng ngày
    daily = (
        temp.groupby(temp['order_date'].dt.date)['sales']
        .sum()
        .reset_index()
    )

    daily.columns = ['date', 'daily_sales']
    daily['date'] = pd.to_datetime(daily['date'])

    # Tạo thứ trong tuần
    day_order = [
        'Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]

    daily['day_of_week'] = daily['date'].dt.day_name()

    fig = px.box(
        daily,
        x='day_of_week',
        y='daily_sales',
        points='all',
        category_orders={'day_of_week': day_order},
        title='📦 Phân phối doanh thu ngày theo thứ trong tuần',
        labels={
            'day_of_week': 'Thứ trong tuần',
            'daily_sales': 'Doanh thu mỗi ngày ($)'
        },
        hover_data=['date']
    )

    fig.update_traces(
        jitter=0.35,
        pointpos=0,
        marker=dict(opacity=0.65, size=5)
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_title='Thứ trong tuần',
        yaxis_title='Doanh thu mỗi ngày ($)'
    )

    return fig


# 2. CALENDAR HEATMAP
# Tháng x Thứ trong tuần
def plot_calendar_heatmap(df):
    """
    Calendar Heatmap:
    - Trục X: Tháng
    - Trục Y: Thứ trong tuần
    - Màu sắc: Tổng doanh thu
    """

    # Tận dụng hàm Heatmap đã có trong dự án
    return plot_sales_heatmap(df)


# 3. BUBBLE SCATTER
# Số giao dịch x AOV, kích thước = doanh thu
def plot_bubble_scatter(df):
    """
    Bubble Scatter:
    - Trục X: Số giao dịch
    - Trục Y: AOV (Average Order Value)
    - Kích thước bong bóng: Tổng doanh thu
    - Mỗi bong bóng: Một danh mục sản phẩm
    """

    if df is None or df.empty:
        return px.scatter(title="⚠️ Không có dữ liệu")

    required = ['sales', 'category']
    if not all(col in df.columns for col in required):
        return px.scatter(
            title="⚠️ Thiếu dữ liệu Sales hoặc Category"
        )

    temp = df.copy()
    temp['sales'] = pd.to_numeric(
        temp['sales'], errors='coerce'
    )
    temp = temp.dropna(subset=['sales', 'category'])

    if temp.empty:
        return px.scatter(title="⚠️ Không có dữ liệu hợp lệ")

    # Số giao dịch duy nhất
    if 'order_id' in temp.columns:
        transaction_count = (
            temp.groupby('category')['order_id'].nunique()
        )
    else:
        transaction_count = temp.groupby('category').size()

    # Tổng doanh thu theo danh mục
    summary = (
        temp.groupby('category')['sales']
        .sum()
        .reset_index(name='total_sales')
    )

    summary['transaction_count'] = (
        summary['category'].map(transaction_count)
    )

    # AOV = Tổng doanh thu / Số giao dịch
    summary['aov'] = (
        summary['total_sales'] /
        summary['transaction_count'].replace(0, float('nan'))
    )

    summary = summary.dropna(
        subset=['transaction_count', 'aov']
    )

    if summary.empty:
        return px.scatter(title="⚠️ Không có dữ liệu để vẽ")

    fig = px.scatter(
        summary,
        x='transaction_count',
        y='aov',
        size='total_sales',
        color='category',
        hover_name='category',
        size_max=60,
        title='🫧 Mối quan hệ giữa số giao dịch và AOV',
        labels={
            'transaction_count': 'Số giao dịch',
            'aov': 'Giá trị đơn hàng trung bình ($)',
            'total_sales': 'Tổng doanh thu ($)',
            'category': 'Danh mục'
        }
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_title='Số giao dịch',
        yaxis_title='AOV ($)'
    )

    return fig


# 4. 100% STACKED BAR
# Kênh phục vụ x Phương thức thanh toán
def plot_100_stacked_bar(df):
    """
    100% Stacked Bar:
    - Trục X: Kênh phục vụ (location)
    - Trục Y: Tỷ lệ giao dịch (%)
    - Màu sắc: Phương thức thanh toán
    """

    if df is None or df.empty:
        return px.bar(title="⚠️ Không có dữ liệu")

    required = ['location', 'payment_method']
    if not all(col in df.columns for col in required):
        return px.bar(
            title="⚠️ Thiếu Location hoặc Payment Method"
        )

    temp = df.copy()

    temp = temp.dropna(
        subset=['location', 'payment_method']
    )

    if temp.empty:
        return px.bar(title="⚠️ Không có dữ liệu hợp lệ")

    # Đếm số giao dịch theo kênh phục vụ và phương thức thanh toán
    if 'order_id' in temp.columns:
        summary = (
            temp.groupby(
                ['location', 'payment_method']
            )['order_id']
            .nunique()
            .reset_index(name='transactions')
        )
    else:
        summary = (
            temp.groupby(
                ['location', 'payment_method']
            )
            .size()
            .reset_index(name='transactions')
        )

    if summary.empty:
        return px.bar(title="⚠️ Không có dữ liệu để vẽ")

    # Tính tỷ lệ phần trăm trong từng kênh phục vụ
    summary['percentage'] = (
        summary['transactions'] /
        summary.groupby('location')['transactions']
        .transform('sum') * 100
    )

    fig = px.bar(
        summary,
        x='location',
        y='percentage',
        color='payment_method',
        barmode='stack',
        text=summary['percentage'].map(
            lambda x: f'{x:.1f}%'
        ),
        title='💳 Tỷ trọng phương thức thanh toán theo kênh phục vụ',
        labels={
            'location': 'Kênh phục vụ',
            'percentage': 'Tỷ lệ giao dịch (%)',
            'payment_method': 'Phương thức thanh toán'
        }
    )

    fig.update_layout(
        template='plotly_white',
        yaxis=dict(
            title='Tỷ lệ giao dịch (%)',
            range=[0, 100],
            ticksuffix='%'
        ),
        xaxis_title='Kênh phục vụ',
        legend_title='Phương thức thanh toán'
    )

    return fig
