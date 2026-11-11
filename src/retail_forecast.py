"""Daily revenue forecasting with chronological backtests."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def daily_revenue(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["order_date"], errors="coerce")
    series = df.assign(order_date=dates).dropna(subset=["order_date"])
    if series.empty:
        raise ValueError("No valid order dates")
    # No recorded order on a calendar date is represented as zero recorded sales.
    return series.groupby("order_date")["sales"].sum().asfreq("D", fill_value=0.0)


def _holt_winters(train: pd.Series, steps: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
        fitted = ExponentialSmoothing(
            train, trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=7, initialization_method="estimated",
        ).fit(optimized=True)
    return np.maximum(0, np.asarray(fitted.forecast(steps), dtype=float))


def _seasonal_naive(train: pd.Series, steps: int) -> np.ndarray:
    return np.resize(train.iloc[-7:].to_numpy(dtype=float), steps)


def evaluate_and_forecast(series: pd.Series, horizon: int = 14, folds: int = 3) -> dict:
    if not 1 <= horizon <= 28:
        raise ValueError("Horizon must be 1–28 days")
    if len(series) < 8 * 7 + horizon * folds:
        raise ValueError("Need more daily history for three backtests")
    evaluations = []
    for fold in range(folds, 0, -1):
        end = len(series) - (fold - 1) * horizon
        train = series.iloc[: end - horizon]
        actual = series.iloc[end - horizon : end].to_numpy(dtype=float)
        for name, prediction in (
            ("Holt-Winters (7 ngày)", _holt_winters(train, horizon)),
            ("Cùng thứ tuần trước", _seasonal_naive(train, horizon)),
        ):
            evaluations.append(pd.DataFrame({
                "fold": folds - fold + 1,
                "date": series.index[end - horizon : end],
                "model": name,
                "actual": actual,
                "prediction": prediction,
            }))
    detail = pd.concat(evaluations, ignore_index=True)
    metrics = detail.assign(abs_error=lambda x: (x.actual - x.prediction).abs()).groupby("model").agg(
        mae=("abs_error", "mean"),
        absolute_error=("abs_error", "sum"),
        actual_total=("actual", "sum"),
    )
    metrics["wape_pct"] = 100 * metrics["absolute_error"] / metrics["actual_total"].replace(0, np.nan)
    metrics = metrics[["mae", "wape_pct"]].sort_values("mae")
    future_dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({
        "date": future_dates,
        "holt_winters": _holt_winters(series, horizon),
        "seasonal_naive": _seasonal_naive(series, horizon),
    })
    return {"metrics": metrics, "detail": detail, "future": future}
