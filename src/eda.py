import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_sales_over_time(df):
    """Biểu đồ xu hướng Doanh thu & Lợi nhuận theo thời gian"""
    df_trend = df.groupby('order_date')[['sales', 'profit']].sum().reset_index()
    fig = px.line(
        df_trend, 
        x='order_date', 
        y=['sales', 'profit'],
        title='📈 Xu hướng Doanh thu & Lợi nhuận theo thời gian',
        labels={'value': 'USD ($)', 'order_date': 'Ngày đặt hàng', 'variable': 'Chỉ số'},
        color_discrete_map={'sales': '#1f77b4', 'profit': '#2ca02c'}
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_category_performance(df):
    """Biểu đồ Doanh thu & Lợi nhuận theo Danh mục sản phẩm (Category)"""
    df_cat = df.groupby('category')[['sales', 'profit']].sum().reset_index()
    fig = px.bar(
        df_cat, 
        x='category', 
        y=['sales', 'profit'],
        barmode='group',
        title='📦 Doanh thu & Lợi nhuận theo Danh mục',
        labels={'value': 'USD ($)', 'category': 'Danh mục'},
        color_discrete_map={'sales': '#1f77b4', 'profit': '#2ca02c'}
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_discount_vs_profit(df):
    """Scatter Plot: Tác động của Chiết khấu (Discount) tới Lợi nhuận (Profit)"""
    fig = px.scatter(
        df, 
        x='discount', 
        y='profit', 
        color='category',
        size='sales',
        hover_data=['sub_category'],
        title='🎯 Mối quan hệ giữa Mức Chiết Khấu & Lợi Nhuận',
        labels={'discount': 'Tỷ lệ chiết khấu (%)', 'profit': 'Lợi nhuận ($)'}
    )
    fig.update_layout(template='plotly_white')
    return fig