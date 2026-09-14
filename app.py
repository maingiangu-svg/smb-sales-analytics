import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans

# 1. Config trang
st.set_page_config(page_title="SMB Analytics Platform", layout="wide")
st.title("📊 SMB Operational Intelligence & Pricing Dashboard")

# 2. Kết nối DB
conn = sqlite3.connect("data.db")


# Hàm load danh sách bảng
def get_tables():
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE"
        " 'sqlite_%'"
    )
    return [row[0] for row in cursor.fetchall()]


tables = get_tables()

if not tables:
    st.error(
        "Chưa có dữ liệu trong SQLite! Hãy chạy `python pipeline.py` trước."
    )
    st.stop()

# 3. Sidebar chọn bảng
selected_table = st.sidebar.selectbox("📂 Chọn dữ liệu (Table)", tables)
df = pd.read_sql_query(f'SELECT * FROM "{selected_table}"', conn)

# Sidebar bộ lọc
st.sidebar.markdown("---")
st.sidebar.write(f"**Tổng số dòng:** {len(df)}")

# 4. Tabs chính của ứng dụng
tab1, tab2, tab3 = st.tabs(
    ["📋 Data Inspector", "📈 EDA & Temporal Trends", "🤖 ML & Pricing Advisor"]
)

# TAB 1: DATA INSPECTOR (Member 2)
with tab1:
    st.subheader("Dữ liệu sau khi chuẩn hóa")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Kiểu dữ liệu các cột:**")
        st.write(df.dtypes.astype(str))
    with col2:
        st.write("**Thống kê Missing Values:**")
        st.write(df.isnull().sum())

# TAB 2: EDA & STATS (Member 3)
with tab2:
    st.subheader("Phân tích Traffic & Xu hướng")

    # Tìm cột ngày tháng
    date_cols = [c for c in df.columns if "date" in c or "ngay" in c]
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if date_cols and num_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])

        # Phân tích theo Thứ trong tuần (Day of Week)
        df["day_of_week"] = df[date_cols[0]].dt.day_name()
        days_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        metric_target = st.selectbox("Chọn chỉ số phân tích:", num_cols)

        dow_summary = (
            df.groupby("day_of_week")[metric_target]
            .sum()
            .reindex(days_order)
            .reset_index()
        )

        fig = px.bar(
            dow_summary,
            x="day_of_week",
            y=metric_target,
            title=f"Tổng {metric_target} theo ngày trong tuần",
            color=metric_target,
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cần ít nhất 1 cột Ngày tháng và 1 cột Số để vẽ chart Trend.")

# TAB 3: ML & PRICING (Member 4)
with tab3:
    st.subheader("Phân khúc Giao dịch / Khách hàng (K-Means)")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(num_cols) >= 2:
        feature_x = st.selectbox("Feature X (Ví dụ: Số lượng/Frequency):", num_cols, index=0)
        feature_y = st.selectbox("Feature Y (Ví dụ: Doanh thu/Monetary):", num_cols, index=min(1, len(num_cols)-1))

        n_clusters = st.slider("Số lượng phân khúc (Clusters):", 2, 5, 3)

        # Fit K-Means
        X = df[[feature_x, feature_y]].dropna()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        X["cluster"] = kmeans.labels_.astype(str)

        fig_cluster = px.scatter(
            X,
            x=feature_x,
            y=feature_y,
            color="cluster",
            title=f"Phân khúc dữ liệu theo {feature_x} và {feature_y}",
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Gợi ý Chiến lược Giá (Pricing Optimization Rule)")
        st.success(
            "📌 **Khuyến nghị:** Các nhóm giao dịch có traffic thấp vào Thứ 3 &"
            " Thứ 4 nên áp dụng mức Discount 10-15% để kích cầu. Tăng nhẹ 5%"
            " giá dịch vụ hot vào ngày cuối tuần."
        )
    else:
        st.warning("Cần ít nhất 2 cột định lượng (số) để chạy ML Clustering.")