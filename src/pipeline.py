import pandas as pd
import numpy as np
import sqlite3
import os

DB_PATH = os.path.join("data", "data.db")

def create_connection():
    """Tạo kết nối tới SQLite Database."""
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def clean_and_transform_data(file_path):
    """
    Đọc file Excel/CSV thô, làm sạch và Feature Engineering:
    - Bỏ dòng rác / Null critical
    - Đổi kiểu dữ liệu chuẩn (Datetime, Numeric)
    - Tính Revenue, Profit Margin, trích xuất biến Thời gian
    """
    # 1. Đọc dữ liệu (Hỗ trợ cả file .xlsx và .csv)
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Chuẩn hóa tên cột (Viết thường, xóa khoảng trắng thừa, thay dấu space/dash bằng '_')
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]

    # Map tên cột Superstore/Excel về chuẩn chung
    column_mapping = {
        'order_date': 'order_date',
        'ship_date': 'ship_date',
        'sales': 'sales',
        'quantity': 'quantity',
        'discount': 'discount',
        'profit': 'profit',
        'category': 'category',
        'sub_category': 'sub_category',
        'region': 'region'
    }
    df.rename(columns=column_mapping, inplace=True)

    # 2. Xử lý định dạng ngày tháng
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        # Bỏ các dòng bị lỗi Order Date
        df = df.dropna(subset=['order_date'])
        
        # Feature Engineering: Bóc tách thời gian
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        df['quarter'] = df['order_date'].dt.quarter
        df['day_of_week'] = df['order_date'].dt.day_name()

    # 3. Ép kiểu dữ liệu số & Điền Null (Loại bỏ ký tự rác như $, phẩy)
    numeric_cols = ['sales', 'quantity', 'discount', 'profit']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Tính toán chỉ số bổ sung
    if 'sales' in df.columns and 'profit' in df.columns:
        # Tránh chia cho 0 khi tính Profit Margin
        df['profit_margin'] = np.where(df['sales'] > 0, (df['profit'] / df['sales']) * 100, 0)

    # Clean trùng lặp (Duplicates)
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