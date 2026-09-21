import os
import sqlite3
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DB_PATH = os.path.join("data", "data.db")
MODEL_PATH = os.path.join("data", "pricing_model.joblib")

def prepare_features(df):
    """
    Chuẩn hóa và trích xuất đặc trưng cho mô hình ML.
    """
    df_clean = df.copy()
    
    # Đảm bảo các cột cần thiết tồn tại
    if 'category' not in df_clean.columns:
        df_clean['category'] = 'General'
    if 'sales' not in df_clean.columns:
        df_clean['sales'] = 0.0
    if 'discount' not in df_clean.columns:
        df_clean['discount'] = 0.0
    if 'quantity' not in df_clean.columns:
        df_clean['quantity'] = 1.0
    if 'profit' not in df_clean.columns:
        df_clean['profit'] = 0.0
        
    # Thêm biến tương tác (Feature Interaction)
    df_clean['sales_x_discount'] = df_clean['sales'] * df_clean['discount']
    
    return df_clean

def build_model_pipeline():
    """
    Xây dựng Pipeline Scikit-Learn gồm tiền xử lý và thuật toán Random Forest.
    """
    categorical_features = ['category']
    numeric_features = ['sales', 'discount', 'quantity', 'sales_x_discount']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, min_samples_leaf=2))
    ])
    
    return pipeline

def train_pricing_model(df=None, model_path=MODEL_PATH):
    """
    Huấn luyện mô hình dự báo Lợi nhuận và lưu ra file joblib.
    """
    if df is None:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Không tìm thấy Database tại {DB_PATH}. Hãy chạy src/pipeline.py trước!")
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM sales_data", conn)
        conn.close()
        
    if df.empty:
        raise ValueError("Dữ liệu rỗng, không thể huấn luyện mô hình!")
        
    df_prepared = prepare_features(df)
    
    feature_cols = ['category', 'sales', 'discount', 'quantity', 'sales_x_discount']
    X = df_prepared[feature_cols]
    y = df_prepared['profit']
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Đánh giá mô hình
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics = {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "num_samples": len(df_prepared)
    }
    
    # Lưu mô hình
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    print(f"[OK] Da huan luyen va luu mo hinh thanh cong tai `{model_path}`!")
    print(f"[METRICS] Ket qua danh gia: R2 = {metrics['r2_score']}, MAE = ${metrics['mae']}, RMSE = ${metrics['rmse']}")
    
    return pipeline, metrics

def load_or_train_model(model_path=MODEL_PATH, db_path=DB_PATH):
    """
    Tải mô hình đã lưu hoặc tự động huấn luyện mới nếu chưa có.
    """
    if os.path.exists(model_path):
        try:
            pipeline = joblib.load(model_path)
            return pipeline
        except Exception as e:
            print(f"[WARNING] Khong the doc file model ({e}), dang huan luyen lai...")
            
    pipeline, _ = train_pricing_model(model_path=model_path)
    return pipeline

def optimize_discount(model, category, expected_sales, max_discount=0.50, num_steps=51):
    """
    Mô phỏng 51 mức chiết khấu từ 0% đến 50% để tìm mức chiết khấu tối ưu lợi nhuận.
    """
    discount_grid = np.linspace(0.0, max_discount, num_steps)
    
    sim_data = pd.DataFrame({
        'category': [category] * num_steps,
        'sales': [expected_sales] * num_steps,
        'discount': discount_grid,
        'quantity': [1.0] * num_steps,
        'sales_x_discount': expected_sales * discount_grid
    })
    
    # Dự báo Lợi nhuận cho từng mức chiết khấu
    predicted_profits = model.predict(sim_data)
    sim_data['predicted_profit'] = predicted_profits
    sim_data['margin_pct'] = np.where(
        expected_sales > 0, 
        (sim_data['predicted_profit'] / expected_sales) * 100, 
        0.0
    )
    
    # Tìm mức chiết khấu có lợi nhuận tối đa
    best_idx = sim_data['predicted_profit'].idxmax()
    best_row = sim_data.loc[best_idx]
    
    result = {
        'optimal_discount': float(best_row['discount']),
        'optimal_discount_pct': round(float(best_row['discount']) * 100, 1),
        'max_profit': float(best_row['predicted_profit']),
        'profit_margin_pct': float(best_row['margin_pct']),
        'simulation_df': sim_data
    }
    
    return result

if __name__ == "__main__":
    print("[INFO] Dang chay script huan luyen mo hinh Machine Learning doc lap...")
    train_pricing_model()
