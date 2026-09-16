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