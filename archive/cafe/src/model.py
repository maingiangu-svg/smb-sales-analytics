import os
import sqlite3
import warnings
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DB_PATH = os.path.join("data", "data.db")
FORECAST_MODEL_PATH = os.path.join("data", "sales_forecast_model.joblib")
PRICING_MODEL_PATH = os.path.join("data", "pricing_model.joblib")


def _month_end_freq():
    try:
        pd.tseries.frequencies.to_offset("ME")
        return "ME"
    except Exception:
        return "M"


# ==============================================================================
# 1. TIME SERIES FORECASTING (HOLT-WINTERS)
# ==============================================================================
def prepare_time_series_data(df, freq=None, category=None):
    if freq is None or freq == "ME":
        freq = _month_end_freq()
    df_ts = df.copy()
    if category and category != "All" and "category" in df_ts.columns:
        df_ts = df_ts[df_ts["category"] == category]

    df_ts["order_date"] = pd.to_datetime(df_ts["order_date"], errors="coerce")
    df_ts = df_ts.dropna(subset=["order_date"]).sort_values("order_date")

    ts_data = df_ts.groupby(pd.Grouper(key="order_date", freq=freq))["sales"].sum().reset_index()
    ts_data = ts_data.set_index("order_date").asfreq(freq, fill_value=0.0)
    return ts_data


def train_forecast_model(df=None, model_path=FORECAST_MODEL_PATH, category=None):
    if df is None:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM sales_data", conn)

    ts_data = prepare_time_series_data(df, category=category)
    test_size = min(3, max(1, len(ts_data) // 4))
    train, test = ts_data.iloc[:-test_size], ts_data.iloc[-test_size:]

    # Với dữ liệu 12 tháng (train = 9 tháng), sử dụng chu kỳ Quý (seasonal_periods=4)
    seasonal_periods = 4 if len(train) >= 8 else max(2, len(train) // 2)
    try:
        model = ExponentialSmoothing(
            train["sales"], trend="add", seasonal="add",
            seasonal_periods=seasonal_periods, initialization_method="estimated"
        ).fit()
        preds = model.forecast(len(test))
        final_model = ExponentialSmoothing(
            ts_data["sales"], trend="add", seasonal="add",
            seasonal_periods=seasonal_periods, initialization_method="estimated"
        ).fit()
    except Exception:
        model = ExponentialSmoothing(train["sales"], trend="add", seasonal=None).fit()
        preds = model.forecast(len(test))
        final_model = ExponentialSmoothing(ts_data["sales"], trend="add", seasonal=None).fit()

    mae = mean_absolute_error(test["sales"], preds)
    rmse = np.sqrt(mean_squared_error(test["sales"], preds))
    metrics = {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2), "num_months": len(ts_data)}

    if not category or category == "All":
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(final_model, model_path)

    return final_model, metrics


def forecast_future_sales(model, steps=6):
    forecast = model.forecast(steps)
    return pd.DataFrame({
        "forecast_date": forecast.index,
        "predicted_sales": np.maximum(0, forecast.values).round(2)
    })


def load_or_train_model(model_path=FORECAST_MODEL_PATH, db_path=DB_PATH):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    model, _ = train_forecast_model(model_path=model_path)
    return model


if __name__ == "__main__":
    _, ts_m = train_forecast_model()
    print(f"✅ Time Series Holt-Winters (Dự báo Doanh thu theo Tháng): MAE = ${ts_m['mae']}, RMSE = ${ts_m['rmse']}")
