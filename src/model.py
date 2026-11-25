import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DB_PATH = os.path.join("data", "data.db")
FORECAST_MODEL_PATH = os.path.join("data", "sales_forecast_model.joblib")
PRICING_MODEL_PATH = os.path.join("data", "pricing_model.joblib")


# ==============================================================================
# 1. TIME SERIES FORECASTING (HOLT-WINTERS)
# ==============================================================================
def prepare_time_series_data(df, freq="D", category=None):
    """Aggregate observed cafe sales by calendar day for a selected category."""
    if freq != "D":
        raise ValueError("Dự báo hiện chỉ hỗ trợ doanh thu theo ngày (freq='D').")
    df_ts = df.copy()
    if category and category != "All" and "category" in df_ts.columns:
        df_ts = df_ts[df_ts["category"] == category]

    df_ts["order_date"] = pd.to_datetime(df_ts["order_date"], errors="coerce")
    df_ts["sales"] = pd.to_numeric(df_ts["sales"], errors="coerce")
    df_ts = df_ts.dropna(subset=["order_date", "sales"])
    if df_ts.empty:
        return pd.DataFrame(columns=["sales"], index=pd.DatetimeIndex([], name="order_date"))

    daily = df_ts.groupby(df_ts["order_date"].dt.normalize())["sales"].sum()
    daily.index.name = "order_date"
    # A date without recorded sales in this category is represented by zero.
    return daily.asfreq("D", fill_value=0.0).to_frame("sales")


def _fit_daily_holt_winters(sales):
    return ExponentialSmoothing(
        sales, trend="add", damped_trend=True, seasonal="add",
        seasonal_periods=7, initialization_method="estimated",
    ).fit(optimized=True)


def train_forecast_model(df=None, model_path=FORECAST_MODEL_PATH, category=None,
                         horizon=14, folds=3, save_model=False):
    """Backtest on consecutive future windows, then fit on all daily history."""
    if df is None:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM sales_data", conn)

    ts_data = prepare_time_series_data(df, category=category)
    if not 1 <= horizon <= 28 or folds < 1:
        raise ValueError("Chọn 1–28 ngày dự báo và ít nhất một lần kiểm tra.")
    if len(ts_data) < 56 + horizon * folds:
        raise ValueError("Cần thêm dữ liệu theo ngày để kiểm tra và học mùa vụ 7 ngày.")

    actuals, predictions, baselines = [], [], []
    for fold in range(folds, 0, -1):
        end = len(ts_data) - (fold - 1) * horizon
        train = ts_data["sales"].iloc[:end - horizon]
        actual = ts_data["sales"].iloc[end - horizon:end].to_numpy(dtype=float)
        fitted = _fit_daily_holt_winters(train)
        predicted = np.maximum(0, np.asarray(fitted.forecast(horizon), dtype=float))
        baseline = np.resize(train.iloc[-7:].to_numpy(dtype=float), horizon)
        actuals.extend(actual)
        predictions.extend(predicted)
        baselines.extend(baseline)

    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)
    baselines = np.asarray(baselines)
    actual_total = actuals.sum()
    metrics = {
        "mae": round(float(mean_absolute_error(actuals, predictions)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actuals, predictions))), 2),
        "wape": round(float(100 * np.abs(actuals - predictions).sum() / actual_total), 1) if actual_total else np.nan,
        "baseline_mae": round(float(mean_absolute_error(actuals, baselines)), 2),
        "baseline_wape": round(float(100 * np.abs(actuals - baselines).sum() / actual_total), 1) if actual_total else np.nan,
        "num_days": len(ts_data),
        "test_days": horizon * folds,
        "folds": folds,
    }

    final_model = _fit_daily_holt_winters(ts_data["sales"])
    if save_model and (not category or category == "All"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(final_model, model_path)

    return final_model, metrics


def forecast_future_sales(model, steps=14):
    forecast = model.forecast(steps)
    return pd.DataFrame({
        "forecast_date": forecast.index,
        "predicted_sales": np.maximum(0, forecast.values).round(2)
    })


# ==============================================================================
# 2. PRICING OPTIMIZATION (BASELINE LR vs ADVANCED RANDOM FOREST)
# ==============================================================================
def train_pricing_models(df=None, model_path=PRICING_MODEL_PATH):
    if df is None:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM sales_data", conn)

    # Baseline: Linear Regression
    X_base = df[["sales", "discount"]].fillna(0)
    y = df["profit"].fillna(0)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_base, y, test_size=0.2, random_state=42)
    lr = LinearRegression().fit(Xb_tr, yb_tr)
    yb_pred = lr.predict(Xb_te)

    # Advanced: Random Forest với Category Encoding
    X_adv = pd.get_dummies(df[["sales", "discount", "category"]], drop_first=False)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_adv, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(Xa_tr, ya_tr)
    ya_pred = rf.predict(Xa_te)

    bundle = {
        "model": rf,
        "features": list(X_adv.columns),
        "baseline_metrics": {
            "r2": round(float(r2_score(yb_te, yb_pred)), 4),
            "mae": round(float(mean_absolute_error(yb_te, yb_pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(yb_te, yb_pred))), 2)
        },
        "advanced_metrics": {
            "r2": round(float(r2_score(ya_te, ya_pred)), 4),
            "mae": round(float(mean_absolute_error(ya_te, ya_pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(ya_te, ya_pred))), 2)
        }
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def optimize_discount(category: str, target_sales: float, model_path=PRICING_MODEL_PATH):
    bundle = joblib.load(model_path) if os.path.exists(model_path) else train_pricing_models(model_path=model_path)
    rf = bundle["model"]
    feat_cols = bundle["features"]

    discounts = np.linspace(0.0, 0.35, 36).round(2)
    sim_df = pd.DataFrame({"sales": target_sales, "discount": discounts, "category": category})
    sim_encoded = pd.get_dummies(sim_df, drop_first=False)
    for col in feat_cols:
        if col not in sim_encoded.columns:
            sim_encoded[col] = 0
    sim_encoded = sim_encoded[feat_cols]

    sim_df["predicted_profit"] = rf.predict(sim_encoded).round(2)
    best_row = sim_df.loc[sim_df["predicted_profit"].idxmax()]
    return {
        "best_discount": float(best_row["discount"]),
        "max_profit": float(best_row["predicted_profit"]),
        "curve": sim_df[["discount", "predicted_profit"]],
        "metrics": bundle["advanced_metrics"]
    }


def load_or_train_model(model_path=FORECAST_MODEL_PATH, db_path=DB_PATH):
    if os.path.exists(model_path):
        saved = joblib.load(model_path)
        index = getattr(saved.model, "_index", None)
        if getattr(index, "freqstr", None) == "D" and getattr(saved.model, "seasonal_periods", None) == 7:
            return saved
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM sales_data", conn)
    model, _ = train_forecast_model(df=df, model_path=model_path)
    return model


if __name__ == "__main__":
    _, ts_m = train_forecast_model()
    p_bundle = train_pricing_models()
    print(f"✅ Time Series Holt-Winters: MAE = ${ts_m['mae']}, RMSE = ${ts_m['rmse']}")
    print(f"✅ Baseline Linear Regression: {p_bundle['baseline_metrics']}")
    print(f"✅ Advanced Random Forest:     {p_bundle['advanced_metrics']}")
