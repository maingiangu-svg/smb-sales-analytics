"""
src/stats.py
 
Kiểm định thống kê: ảnh hưởng của mức Chiết khấu (Discount) tới Lợi nhuận (Profit).
 
Luồng dùng:
    - Từ dashboard:  render_streamlit(df)
    - Lấy số liệu:   run_all(df) -> dict gồm summary, ttest, anova, conclusion
 
Chỉ phụ thuộc pandas, numpy, scipy (streamlit/plotly chỉ import khi vẽ dashboard).
"""
from __future__ import annotations
 
import sqlite3
 
import numpy as np
import pandas as pd
from scipy import stats as sps
 
ALPHA = 0.05
DISCOUNT_COL = "discount"
PROFIT_COL = "profit"
GROUP_COL = "Discount_Group"
 
# Discount dạng tỉ lệ (0, 0.1, 0.2, ...). Khoảng đóng bên phải: (a, b]
_BINS = [-np.inf, 0, 0.2, 0.4, np.inf]
_LABELS = ["Không giảm giá", "Giảm ≤20%", "Giảm 20-40%", "Giảm >40%"]
 
 
# --------------------------------------------------------------------------
# Đọc và chuẩn bị dữ liệu
# --------------------------------------------------------------------------
def load_from_sqlite(db_path: str, table: str) -> pd.DataFrame:
    """Đọc nguyên một bảng từ SQLite (dùng khi chạy độc lập, ngoài dashboard)."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
 
 
def prepare(
    df: pd.DataFrame,
    discount_col: str = DISCOUNT_COL,
    profit_col: str = PROFIT_COL,
) -> pd.DataFrame:
    """Giữ 2 cột cần thiết, ép kiểu số, bỏ dòng thiếu, gán nhóm chiết khấu."""
    data = df[[discount_col, profit_col]].copy()
    data.columns = [DISCOUNT_COL, PROFIT_COL]
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
 
    # Nếu Discount lưu dạng phần trăm (0-100) thì đổi về tỉ lệ (0-1)
    if not data.empty and data[DISCOUNT_COL].max() > 1:
        data[DISCOUNT_COL] = data[DISCOUNT_COL] / 100
 
    data[GROUP_COL] = pd.cut(data[DISCOUNT_COL], bins=_BINS, labels=_LABELS)
    return data
 
 
# --------------------------------------------------------------------------
# Thống kê mô tả theo nhóm
# --------------------------------------------------------------------------
def group_summary(data: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Số đơn, lợi nhuận TB, độ lệch chuẩn, trung vị và CI của trung bình theo nhóm."""
    rows = []
    for name, g in data.groupby(GROUP_COL, observed=True):
        x = g[PROFIT_COL]
        n = len(x)
        mean = x.mean()
        if n > 1:
            sem = x.std(ddof=1) / np.sqrt(n)
            margin = sps.t.ppf(1 - alpha / 2, n - 1) * sem
        else:
            margin = np.nan
        rows.append(
            {
                "Nhóm": name,
                "Số đơn": n,
                "Lợi nhuận TB": mean,
                "Trung vị": x.median(),
                "Độ lệch chuẩn": x.std(ddof=1) if n > 1 else np.nan,
                "CI dưới": mean - margin,
                "CI trên": mean + margin,
            }
        )
    return pd.DataFrame(rows)
 
 
# --------------------------------------------------------------------------
# T-test: có chiết khấu vs không chiết khấu
# --------------------------------------------------------------------------
def _welch_ci(a: pd.Series, b: pd.Series, alpha: float = ALPHA):
    """Chênh lệch trung bình (a - b) và CI theo Welch."""
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se2 = v1 / n1 + v2 / n2
    dof = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    diff = a.mean() - b.mean()
    margin = sps.t.ppf(1 - alpha / 2, dof) * np.sqrt(se2)
    return diff, diff - margin, diff + margin
 
 
def ttest_discount(data: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """Welch t-test (không giả định phương sai bằng nhau) + Mann-Whitney để đối chiếu."""
    no_disc = data.loc[data[DISCOUNT_COL] == 0, PROFIT_COL]
    with_disc = data.loc[data[DISCOUNT_COL] > 0, PROFIT_COL]
    if len(no_disc) < 2 or len(with_disc) < 2:
        raise ValueError("Cần ít nhất 2 đơn ở cả nhóm có và không có chiết khấu.")
 
    t_stat, p_value = sps.ttest_ind(with_disc, no_disc, equal_var=False)
    diff, ci_low, ci_high = _welch_ci(with_disc, no_disc, alpha)
 
    n1, n2 = len(with_disc), len(no_disc)
    pooled_sd = np.sqrt(
        ((n1 - 1) * with_disc.var(ddof=1) + (n2 - 1) * no_disc.var(ddof=1)) / (n1 + n2 - 2)
    )
    cohen_d = diff / pooled_sd if pooled_sd > 0 else np.nan
 
    _, p_mw = sps.mannwhitneyu(with_disc, no_disc, alternative="two-sided")
 
    return {
        "n_discount": n1,
        "n_no_discount": n2,
        "mean_discount": with_disc.mean(),
        "mean_no_discount": no_disc.mean(),
        "mean_diff": diff,  # có chiết khấu - không chiết khấu
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohen_d": cohen_d,
        "p_mannwhitney": p_mw,
        "significant": bool(p_value < alpha),
    }
 
 
# --------------------------------------------------------------------------
# ANOVA: so sánh lợi nhuận giữa các mức chiết khấu
# --------------------------------------------------------------------------
def anova_discount_groups(data: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """One-way ANOVA + Kruskal-Wallis + Levene; Tukey HSD nếu ANOVA có ý nghĩa."""
    groups = {
        name: g[PROFIT_COL].to_numpy()
        for name, g in data.groupby(GROUP_COL, observed=True)
        if len(g) >= 2
    }
    if len(groups) < 2:
        raise ValueError("Cần ít nhất 2 nhóm chiết khấu có đủ dữ liệu để chạy ANOVA.")
 
    samples = list(groups.values())
    f_stat, p_value = sps.f_oneway(*samples)
    h_stat, p_kruskal = sps.kruskal(*samples)
    _, p_levene = sps.levene(*samples)
 
    all_vals = np.concatenate(samples)
    grand_mean = all_vals.mean()
    ss_between = sum(len(v) * (v.mean() - grand_mean) ** 2 for v in samples)
    ss_total = ((all_vals - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
 
    posthoc = None
    if p_value < alpha:
        try:
            tukey = sps.tukey_hsd(*samples)
            names = list(groups.keys())
            rows = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    rows.append(
                        {
                            "Nhóm A": names[i],
                            "Nhóm B": names[j],
                            "Chênh lệch TB (A - B)": samples[i].mean() - samples[j].mean(),
                            "p (Tukey)": tukey.pvalue[i, j],
                        }
                    )
            posthoc = pd.DataFrame(rows)
        except AttributeError:  # scipy cũ chưa có tukey_hsd
            posthoc = None
 
    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "eta_squared": eta_sq,
        "h_kruskal": h_stat,
        "p_kruskal": p_kruskal,
        "p_levene": p_levene,
        "equal_variance": bool(p_levene >= alpha),
        "significant": bool(p_value < alpha),
        "posthoc": posthoc,
    }
 
 
# --------------------------------------------------------------------------
# Kết luận kinh doanh
# --------------------------------------------------------------------------
def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"
 
 
def business_conclusion(ttest: dict, anova: dict, alpha: float = ALPHA) -> str:
    """Kết luận ngắn gọn, đọc được bởi người không chuyên thống kê."""
    diff = ttest["mean_diff"]
    ci = f"[{ttest['ci_low']:,.2f}; {ttest['ci_high']:,.2f}]"
    lines = []
 
    if ttest["significant"]:
        direction = "thấp hơn" if diff < 0 else "cao hơn"
        lines.append(
            f"Đơn có chiết khấu có lợi nhuận trung bình {direction} {abs(diff):,.2f} "
            f"so với đơn không chiết khấu (CI {int((1 - alpha) * 100)}%: {ci}, "
            f"p = {_fmt_p(ttest['p_value'])}). Khác biệt có ý nghĩa thống kê."
        )
        if diff < 0:
            lines.append("Chiết khấu đang kéo lợi nhuận xuống, nên cân nhắc đặt trần chiết khấu.")
    else:
        lines.append(
            f"Chưa đủ bằng chứng cho thấy chiết khấu làm thay đổi lợi nhuận trung bình "
            f"(chênh lệch {diff:,.2f}, CI {ci}, p = {_fmt_p(ttest['p_value'])})."
        )
 
    if anova["significant"]:
        lines.append(
            f"Lợi nhuận khác nhau giữa các mức chiết khấu (ANOVA p = {_fmt_p(anova['p_value'])}, "
            f"eta² = {anova['eta_squared']:.3f}); xem bảng so sánh cặp để biết mức nào khác biệt."
        )
    else:
        lines.append(
            f"Không thấy khác biệt có ý nghĩa giữa các mức chiết khấu "
            f"(ANOVA p = {_fmt_p(anova['p_value'])})."
        )
 
    lines.append("Lưu ý: đây là mối quan hệ thống kê, chưa chứng minh chiết khấu là nguyên nhân.")
    return " ".join(lines)
 
 
# --------------------------------------------------------------------------
# Chạy toàn bộ
# --------------------------------------------------------------------------
def run_all(
    df: pd.DataFrame,
    discount_col: str = DISCOUNT_COL,
    profit_col: str = PROFIT_COL,
    alpha: float = ALPHA,
) -> dict:
    data = prepare(df, discount_col, profit_col)
    ttest = ttest_discount(data, alpha)
    anova = anova_discount_groups(data, alpha)
    return {
        "summary": group_summary(data, alpha),
        "ttest": ttest,
        "anova": anova,
        "conclusion": business_conclusion(ttest, anova, alpha),
    }
 
 
# --------------------------------------------------------------------------
# Hiển thị trên Streamlit
# --------------------------------------------------------------------------
def render_streamlit(
    df: pd.DataFrame,
    discount_col: str = DISCOUNT_COL,
    profit_col: str = PROFIT_COL,
    alpha: float = ALPHA,
) -> None:
    """Vẽ kết quả kiểm định lên dashboard. Gọi từ app.py: stats.render_streamlit(df)."""
    import plotly.express as px
    import streamlit as st
 
    st.subheader("Kiểm định thống kê: Chiết khấu và Lợi nhuận")
 
    try:
        res = run_all(df, discount_col, profit_col, alpha)
    except (KeyError, ValueError) as e:
        st.warning(f"Không chạy được kiểm định: {e}")
        return
 
    t, a = res["ttest"], res["anova"]
 
    st.info(res["conclusion"])
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Chênh lệch lợi nhuận TB (có - không CK)", f"{t['mean_diff']:,.2f}")
    c2.metric("p-value (t-test)", _fmt_p(t["p_value"]))
    c3.metric(
        f"CI {int((1 - alpha) * 100)}% của chênh lệch",
        f"{t['ci_low']:,.2f} → {t['ci_high']:,.2f}",
    )
 
    summary = res["summary"].copy()
    summary["Sai số"] = summary["CI trên"] - summary["Lợi nhuận TB"]
    fig = px.bar(
        summary,
        x="Nhóm",
        y="Lợi nhuận TB",
        error_y="Sai số",
        text_auto=".2f",
        title="Lợi nhuận trung bình theo mức chiết khấu (thanh sai số = CI)",
    )
    st.plotly_chart(fig, width="stretch")
 
    st.markdown("**Thống kê theo nhóm**")
    st.dataframe(res["summary"], width="stretch")
 
    with st.expander("Chi tiết kiểm định"):
        st.write(
            {
                "Welch t-test": {
                    "t": round(t["t_stat"], 4),
                    "p": t["p_value"],
                    "Cohen's d": round(t["cohen_d"], 4),
                    "Mann-Whitney p": t["p_mannwhitney"],
                },
                "ANOVA": {
                    "F": round(a["f_stat"], 4),
                    "p": a["p_value"],
                    "eta²": round(a["eta_squared"], 4),
                    "Kruskal-Wallis p": a["p_kruskal"],
                    "Levene p (phương sai đều)": a["p_levene"],
                },
            }
        )
        if a["posthoc"] is not None:
            st.markdown("**So sánh cặp (Tukey HSD)**")
            st.dataframe(a["posthoc"], width="stretch")
 
