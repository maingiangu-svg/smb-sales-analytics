"""Retail revenue and profit analysis on observed Sample Superstore data."""

import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.retail_analysis import discount_regression, discount_summary, performance
from src.retail_forecast import daily_revenue, evaluate_and_forecast
from src.retail_pipeline import DB_PATH, RAW_PATH, TABLE, audit_raw, read_raw


st.set_page_config(page_title="Phân tích bán lẻ Superstore", page_icon="📊", layout="wide")


@st.cache_data
def load_retail() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f'SELECT * FROM "{TABLE}"', conn)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["is_loss"] = df["is_loss"].astype(bool)
    return df


@st.cache_data
def load_audit() -> dict:
    return audit_raw(read_raw(RAW_PATH))


@st.cache_data(show_spinner="Đang đánh giá mô hình theo từng mốc thời gian...")
def forecast_for_scope(df: pd.DataFrame, days: int) -> dict:
    return evaluate_and_forecast(daily_revenue(df), horizon=days)


st.title("Phân tích doanh thu và lợi nhuận bán lẻ")
st.caption("Sample Superstore · doanh thu, lợi nhuận và chiết khấu là số liệu có sẵn trong nguồn")

if not DB_PATH.exists():
    st.error("Chưa có cơ sở dữ liệu bán lẻ. Chạy `python3 -m src.retail_pipeline` trước.")
    st.stop()

df = load_retail()
if df.empty:
    st.error("Bảng giao dịch bán lẻ đang trống.")
    st.stop()

regions = sorted(df["region"].dropna().unique().tolist())
categories = sorted(df["category"].dropna().unique().tolist())
years = sorted(df["year"].unique().tolist())
with st.sidebar:
    st.header("Bộ lọc")
    chosen_regions = st.multiselect("Khu vực", regions, default=regions)
    chosen_categories = st.multiselect("Danh mục", categories, default=categories)
    chosen_years = st.multiselect("Năm (phân tích)", years, default=years)
    st.caption("Dự báo dùng toàn bộ lịch sử của khu vực và danh mục đã chọn.")

scope = df[df["region"].isin(chosen_regions) & df["category"].isin(chosen_categories)]
view = scope[scope["year"].isin(chosen_years)]
if view.empty:
    st.warning("Bộ lọc hiện tại không có giao dịch.")
    st.stop()

tab_overview, tab_discount, tab_forecast, tab_quality = st.tabs([
    "Tổng quan & EDA", "Chiết khấu & hồi quy", "Dự báo theo ngày", "Chất lượng dữ liệu",
])

with tab_overview:
    sales, profit = view["sales"].sum(), view["profit"].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Doanh thu", f"${sales:,.0f}")
    k2.metric("Lợi nhuận", f"${profit:,.0f}")
    k3.metric("Biên lợi nhuận", f"{100 * profit / sales:.1f}%")
    k4.metric("Đơn hàng", f"{view['order_id'].nunique():,}")
    st.caption("Doanh thu và lợi nhuận là tổng các dòng sản phẩm; số đơn là Order ID duy nhất.")

    daily = daily_revenue(view).rename("sales").reset_index().rename(columns={"order_date": "date"})
    st.plotly_chart(px.line(daily, x="date", y="sales", title="Doanh thu ghi nhận theo ngày"), width="stretch")
    left, right = st.columns(2)
    with left:
        region = performance(view, "region")
        st.plotly_chart(px.bar(region, x="region", y=["sales", "profit"], barmode="group",
                               title="Doanh thu và lợi nhuận theo khu vực"), width="stretch")
        st.dataframe(region.style.format({"sales": "${:,.0f}", "profit": "${:,.0f}",
                                          "margin_pct": "{:.1f}%"}), hide_index=True, width="stretch")
    with right:
        subcat = performance(view, "sub_category")
        st.plotly_chart(px.bar(subcat.sort_values("profit").head(10), x="profit", y="sub_category",
                               orientation="h", color="profit", color_continuous_scale="RdBu",
                               title="10 nhóm sản phẩm có lợi nhuận thấp nhất"), width="stretch")
        st.dataframe(subcat[["sub_category", "sales", "profit", "margin_pct"]].style.format(
            {"sales": "${:,.0f}", "profit": "${:,.0f}", "margin_pct": "{:.1f}%"}),
            hide_index=True, width="stretch")

with tab_discount:
    st.subheader("Chiết khấu và lợi nhuận")
    bands = discount_summary(view)
    st.plotly_chart(px.bar(bands, x="discount_band", y="margin_pct", color="loss_rate",
                           labels={"discount_band": "Mức chiết khấu", "margin_pct": "Biên lợi nhuận (%)",
                                   "loss_rate": "Tỷ lệ dòng lỗ"},
                           title="Biên lợi nhuận theo mức chiết khấu"), width="stretch")
    st.dataframe(bands.style.format({"sales": "${:,.0f}", "profit": "${:,.0f}",
                                      "avg_profit": "${:,.2f}", "loss_rate": "{:.1%}",
                                      "margin_pct": "{:.1f}%"}), hide_index=True, width="stretch")
    st.caption("Các nhóm chiết khấu là thống kê mô tả trên từng dòng sản phẩm, không phải thử nghiệm A/B.")
    try:
        regression = discount_regression(view)
        st.markdown("#### Hồi quy biên lợi nhuận")
        a, b, c = st.columns(3)
        a.metric("Thay đổi ước lượng khi chiết khấu tăng 10 điểm %",
                 f"{regression['per_10pp']:+.1f} điểm %")
        b.metric("Khoảng tin cậy 95%", f"[{regression['ci_low']:+.1f}; {regression['ci_high']:+.1f}]")
        c.metric("R² / số dòng", f"{regression['r2']:.2f} / {regression['n']:,}")
        st.caption("OLS với sai số chuẩn HC3; kiểm soát quy mô doanh thu (log), số lượng, danh mục, khu vực và năm. "
                   "Đây là mối liên hệ trong dữ liệu quan sát, không chứng minh chiết khấu gây ra thay đổi lợi nhuận.")
    except ValueError as exc:
        st.info(str(exc))

with tab_forecast:
    st.subheader("Dự báo doanh thu theo ngày")
    days = st.slider("Số ngày dự báo", min_value=7, max_value=28, value=14, step=7)
    st.caption("Mùa vụ theo tuần (7 ngày). Ngày không có giao dịch ghi nhận được tính là 0; "
               "không thể xác minh từ nguồn liệu cửa hàng có đóng cửa vào những ngày đó hay không.")
    try:
        results = forecast_for_scope(scope, days)
        metrics = results["metrics"]
        st.markdown(f"**Đánh giá trên 3 khoảng liên tiếp, mỗi khoảng {days} ngày cuối lịch sử**")
        st.dataframe(metrics.style.format({"mae": "${:,.0f}", "wape_pct": "{:.1f}%"}), width="stretch")
        st.caption("MAE là sai số tuyệt đối trung bình mỗi ngày; WAPE = tổng sai số tuyệt đối / tổng doanh thu thực tế. "
                   "Các mô hình chỉ học từ dữ liệu trước mỗi khoảng kiểm tra.")
        series = daily_revenue(scope)
        future = results["future"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.tail(90).index, y=series.tail(90).values,
                                 name="Thực tế", mode="lines"))
        fig.add_trace(go.Scatter(x=future["date"], y=future["holt_winters"],
                                 name="Holt-Winters", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=future["date"], y=future["seasonal_naive"],
                                 name="Cùng thứ tuần trước", mode="lines", line={"dash": "dot"}))
        fig.update_layout(title="90 ngày cuối và dự báo tiếp theo", xaxis_title="Ngày",
                          yaxis_title="Doanh thu (USD)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(future.style.format({"holt_winters": "${:,.0f}",
                                          "seasonal_naive": "${:,.0f}"}),
                     hide_index=True, width="stretch")
        if metrics.loc["Holt-Winters (7 ngày)", "mae"] >= metrics.loc["Cùng thứ tuần trước", "mae"]:
            st.warning("Trong các khoảng kiểm tra này, Holt-Winters chưa vượt cách dự báo cùng thứ tuần trước.")
    except ValueError as exc:
        st.info(str(exc))

with tab_quality:
    st.subheader("Nguồn và kiểm toán dữ liệu")
    audit = load_audit()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dòng nguồn", f"{audit['rows']:,}")
    c2.metric("Dòng phân tích", f"{len(df):,}")
    c3.metric("Ô thiếu nguồn", f"{audit['missing_cells']:,}")
    c4.metric("Row ID trùng", f"{audit['duplicate_row_ids']:,}")
    st.write("Ô thiếu theo cột:", audit["missing_by_column"] or "Không có")
    st.write("Ngày đầu:", df["order_date"].min().date(), "· Ngày cuối:", df["order_date"].max().date())
    st.info("File archive.zip được thầy gợi ý chỉ có Sales. Dự án dùng bản Sample Superstore "
            "21 cột có Profit và Discount quan sát được. Dữ liệu là mẫu công khai để học tập, "
            "không đại diện cho hoạt động của một doanh nghiệp cụ thể.")
