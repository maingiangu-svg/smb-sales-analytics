import pandas as pd
import numpy as np
import sqlite3
import os
import re

DB_PATH = os.path.join("data", "data.db")

def normalize_headers(columns):
    """Chuẩn hóa tên cột dạng SQL-friendly"""
    return [
        re.sub(r'[^a-zA-Z0-9]+', '_', str(c).strip().lower()).strip('_') 
        for c in columns
    ]

def create_connection():
    """Tạo kết nối tới SQLite Database."""
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def clean_and_transform_data(file_path):
    # 1. Đọc dữ liệu
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding_errors='ignore')
    else:
        df = pd.read_excel(file_path)

    # Chuẩn hóa tên cột: Chữ thường, xóa khoảng trắng thừa
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]

    # Map chính xác theo từ khóa bất kể vị trí
    col_map = {}
    for col in df.columns:
        if 'profit' in col and 'margin' not in col:
            col_map[col] = 'profit'
        elif 'discount' in col or 'disc' in col:
            col_map[col] = 'discount'
        elif 'sales' in col or 'revenue' in col:
            col_map[col] = 'sales'

    if col_map:
        df.rename(columns=col_map, inplace=True)

    # Giả lập dữ liệu Profit & Discount nếu file gốc thiếu (để hiển thị KPI trên UI)
    np.random.seed(42)
    if 'profit' not in df.columns:
        if 'sales' in df.columns:
            # Profit giả lập từ -10% đến +25% Doanh thu
            df['profit'] = df['sales'] * np.random.uniform(-0.10, 0.25, size=len(df))
        else:
            df['profit'] = 0.0

    if 'discount' not in df.columns:
        # Discount giả lập từ 0% đến 20%
        df['discount'] = np.random.choice([0.0, 0.05, 0.1, 0.15, 0.2], size=len(df))

    # 2. Xử lý định dạng ngày tháng
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        df['quarter'] = df['order_date'].dt.quarter
        df['day_of_week'] = df['order_date'].dt.day_name()

    # 3. Ép kiểu dữ liệu số & làm sạch ký tự rác
    numeric_cols = ['sales', 'quantity', 'discount', 'profit']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = (
                    df[col].astype(str)
                    .str.replace('$', '', regex=False)
                    .str.replace('%', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .str.replace('(', '-', regex=False)
                    .str.replace(')', '', regex=False)
                    .str.strip()
                )
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Tính toán chỉ số bổ sung
    if 'sales' in df.columns and 'profit' in df.columns:
        df['profit_margin'] = np.where(df['sales'] > 0, (df['profit'] / df['sales']) * 100, 0)

    df = df.drop_duplicates()
    return df

def run_pipeline(excel_file_path):
    """
    Chạy toàn bộ Pipeline: Clean -> Lưu vào SQLite
    """
    print(f"🔄 Đang xử lý file: {excel_file_path}...")
    df_clean = clean_and_transform_data(excel_file_path)
    
    conn = create_connection()
    # Ghi dữ liệu vào bảng 'sales_data', nếu tồn tại rồi thì ghi đè (replace)
    df_clean.to_sql("sales_data", conn, if_exists="replace", index=False)
    conn.close()
    
    print(f"✅ Đã xử lý xong {len(df_clean)} dòng dữ liệu và lưu vào SQLite ({DB_PATH})!")
    return df_clean

if __name__ == "__main__":
    sample_file = os.path.join("data", "train.csv")
    if os.path.exists(sample_file):
        run_pipeline(sample_file)
    else:
        print("💡 Chưa có file mẫu trong data/. Hãy thả 1 file Excel/CSV vào thư mục data/ để test.")