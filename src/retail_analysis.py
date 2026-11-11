"""Descriptive retail results and an observational discount regression."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DISCOUNT_BANDS = ["0%", ">0–20%", ">20–40%", ">40–60%", ">60%"]


def performance(df: pd.DataFrame, by: str) -> pd.DataFrame:
    result = df.groupby(by, dropna=False).agg(
        sales=("sales", "sum"),
        profit=("profit", "sum"),
        order_lines=("row_id", "count"),
        loss_lines=("is_loss", "sum"),
    ).reset_index()
    result["margin_pct"] = 100 * result["profit"] / result["sales"]
    return result.sort_values("profit", ascending=False)


def discount_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.groupby("discount_band").agg(
        order_lines=("row_id", "count"),
        sales=("sales", "sum"),
        profit=("profit", "sum"),
        avg_profit=("profit", "mean"),
        loss_rate=("is_loss", "mean"),
    ).reindex(DISCOUNT_BANDS).dropna(subset=["order_lines"])
    result["margin_pct"] = 100 * result["profit"] / result["sales"]
    return result.reset_index()


def discount_regression(df: pd.DataFrame) -> dict:
    """OLS with HC3 errors; association, not a causal discount effect.

    Profit margin is measured in percentage points; discount is a fraction.
    Category, region and year are controls for observed group differences.
    """
    needed = ["profit_margin", "discount", "sales", "quantity", "category", "region", "year"]
    data = df[needed].dropna().copy()
    if len(data) < 100 or data["discount"].nunique() < 2:
        raise ValueError("Not enough varied transactions for the regression")
    data["log_sales"] = np.log1p(data["sales"])
    model = smf.ols(
        "profit_margin ~ discount + log_sales + quantity + C(category) + C(region) + C(year)",
        data=data,
    ).fit(cov_type="HC3")
    low, high = model.conf_int().loc["discount"]
    return {
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "per_10pp": float(model.params["discount"] * 0.1),
        "ci_low": float(low * 0.1),
        "ci_high": float(high * 0.1),
        "p_value": float(model.pvalues["discount"]),
    }
