
import pandas as pd
import streamlit as st


def render_business_report(df):
    """
    Hiển thị Business Report từ dữ liệu bán hàng.
    """

    if df is None or df.empty:
        st.warning("⚠️ Không có dữ liệu để tạo Business Report.")
        return

    st.subheader("📑 Business Report")
    st.caption("Tổng hợp tình hình kinh doanh từ dữ liệu bán hàng.")

    # =========================
    # 1. KPI
    # =========================

    total_sales = (
        df["sales"].sum()
        if "sales" in df.columns
        else 0
    )

    total_profit = (
        df["profit"].sum()
        if "profit" in df.columns
        else 0
    )

    total_orders = (
        df["order_id"].nunique()
        if "order_id" in df.columns
        else len(df)
    )

    avg_discount = (
        df["discount"].mean() * 100
        if "discount" in df.columns
        else 0
    )

    profit_margin = (
        (total_profit / total_sales) * 100
        if total_sales != 0
        else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "💵 Tổng doanh thu",
        f"${total_sales:,.2f}"
    )

    col2.metric(
        "💰 Tổng lợi nhuận",
        f"${total_profit:,.2f}"
    )

    col3.metric(
        "📦 Tổng đơn hàng",
        f"{total_orders:,}"
    )

    col4.metric(
        "🏷️ Chiết khấu TB",
        f"{avg_discount:.1f}%"
    )

    col5.metric(
        "📈 Profit Margin",
        f"{profit_margin:.1f}%"
    )

    st.markdown("---")

    # =========================
    # 2. Hiệu quả theo Category
    # =========================

    if "category" in df.columns:

        st.subheader("📦 Hiệu quả kinh doanh theo Category")

        category_report = (
            df.groupby("category")
            .agg(
                Revenue=("sales", "sum"),
                Profit=("profit", "sum"),
                Orders=("order_id", "nunique")
                if "order_id" in df.columns
                else ("sales", "count")
            )
            .reset_index()
        )

        category_report["Profit Margin (%)"] = (
            category_report["Profit"]
            / category_report["Revenue"]
            * 100
        ).fillna(0)

        category_report = category_report.sort_values(
            "Revenue",
            ascending=False
        )

        st.dataframe(
            category_report,
            width="stretch",
            hide_index=True
        )

    # =========================
    # 3. Top sản phẩm
    # =========================

    subcategory_col = None

    for col in ["sub_category", "subcategory", "sub-category"]:
        if col in df.columns:
            subcategory_col = col
            break

    if subcategory_col and "profit" in df.columns:

        st.subheader("🏆 Top 10 Sub-category theo lợi nhuận")

        top_subcategory = (
            df.groupby(subcategory_col)["profit"]
            .sum()
            .reset_index()
            .sort_values("profit", ascending=False)
            .head(10)
        )

        top_subcategory.columns = [
            "Sub-category",
            "Total Profit"
        ]

        st.dataframe(
            top_subcategory,
            width="stretch",
            hide_index=True
        )

    # =========================
    # 4. Báo cáo theo thời gian
    # =========================

    if "order_date" in df.columns:

        st.subheader("📅 Báo cáo kinh doanh theo tháng")

        report_time = df.copy()

        report_time["order_date"] = pd.to_datetime(
            report_time["order_date"],
            errors="coerce"
        )

        report_time = report_time.dropna(
            subset=["order_date"]
        )

        if not report_time.empty:

            monthly = (
                report_time
                .set_index("order_date")
                .resample("ME")
                .agg({
                    "sales": "sum",
                    "profit": "sum"
                })
                .reset_index()
            )

            monthly["Month"] = monthly["order_date"].dt.strftime(
                "%Y-%m"
            )

            monthly = monthly[
                ["Month", "sales", "profit"]
            ]

            monthly.columns = [
                "Month",
                "Revenue",
                "Profit"
            ]

            st.dataframe(
                monthly,
                width="stretch",
                hide_index=True
            )

    # =========================
    # 5. Business Summary
    # =========================

    st.markdown("---")
    st.subheader("📝 Business Summary")

    if total_sales > 0:
        st.write(
            f"- Tổng doanh thu đạt **${total_sales:,.2f}**."
        )

    if total_profit >= 0:
        st.write(
            f"- Tổng lợi nhuận đạt **${total_profit:,.2f}**."
        )
    else:
        st.write(
            f"- Tổng lợi nhuận đang âm **${total_profit:,.2f}**."
        )

    st.write(
        f"- Tổng số đơn hàng: **{total_orders:,}**."
    )

    st.write(
        f"- Mức chiết khấu trung bình: **{avg_discount:.1f}%**."
    )

    st.write(
        f"- Biên lợi nhuận: **{profit_margin:.1f}%**."
    )
