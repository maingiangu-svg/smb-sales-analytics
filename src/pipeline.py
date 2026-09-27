import os
import re
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = os.path.join("data", "data.db")
TABLE_NAME = "sales_data"


def normalize_headers(columns):
    """Chuẩn hóa tên cột dạng SQL-friendly."""
    return [
        re.sub(r'[^a-zA-Z0-9]+', '_', str(c).strip().lower()).strip('_')
        for c in columns
    ]


def create_connection():
    """Tạo kết nối tới SQLite Database."""
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


# ==============================================================================
# MODULE 1: DATA QUALITY & PROCESSING (Bài 3)
# ==============================================================================
def process_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý cấu trúc bảng, kiểu dữ liệu, Missing Values và Duplicates."""
    df = df.copy()

    # 1.1 Chuẩn hóa Header
    df.columns = normalize_headers(df.columns)

    # 1.2 Map từ khóa cột chuẩn
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

    # 1.3 Khử trùng lặp (Duplicates)
    df = df.drop_duplicates()

    # 1.4 Làm sạch ký tự tiền tệ, phần trăm & ép kiểu số
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
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1.5 Xử lý Missing Values
    if 'order_id' in df.columns:
        df = df.dropna(subset=['order_id'])
    
    cat_cols = ['category', 'sub_category', 'segment', 'region', 'ship_mode']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


# ==============================================================================
# MODULE 2: OUTLIERS, NOISE & CONSISTENCY (Bài 4)
# ==============================================================================
def handle_outliers_noise_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tính nhất quán văn bản, lọc nhiễu logic và giới hạn ngoại lai."""
    df = df.copy()

    # 2.1 Consistency: Đồng nhất định dạng chuỗi phân loại (bỏ khoảng trắng thừa, Title Case)
    text_cols = ['category', 'sub_category', 'segment', 'region', 'customer_name']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # 2.2 Noise Filtering: Lọc bỏ đơn hàng có giá trị vô lý hoặc sai lệch ngày tháng
    if 'sales' in df.columns:
        df = df[df['sales'] > 0]

    # Ép kiểu ngày tháng kiểm tra logic giao vận
    date_cols = ['order_date', 'ship_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'order_date' in df.columns:
        df = df.dropna(subset=['order_date'])

    if 'order_date' in df.columns and 'ship_date' in df.columns:
        # Loại bỏ bản ghi có ngày giao trước ngày đặt (Nhiễu nhập liệu)
        valid_dates = (df['ship_date'].isna()) | (df['ship_date'] >= df['order_date'])
        df = df[valid_dates]

    # 2.3 Outliers Handling: Áp dụng IQR Capping (Winsorization) cho cột 'sales'
    if 'sales' in df.columns and len(df) > 0:
        q1 = df['sales'].quantile(0.25)
        q3 = df['sales'].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 3.0 * iqr  # Ngưỡng 3*IQR bảo tồn các đơn hàng bán buôn lớn
        df['sales'] = np.where(df['sales'] > upper_limit, upper_limit, df['sales'])

    return df


# ==============================================================================
# MODULE 3: TRANSFORMATION, REDUCTION & INTEGRATION (Bài 5)
# ==============================================================================
def transform_and_reduce(df: pd.DataFrame) -> pd.DataFrame:
    """Biến đổi thuộc tính, áp dụng Heuristic Imputation và thu giảm chiều dữ liệu."""
    df = df.copy()

    # 3.1 Feature Extraction từ ngày tháng
    if 'order_date' in df.columns:
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        df['quarter'] = df['order_date'].dt.quarter
        df['day_of_week'] = df['order_date'].dt.day_name()

    # 3.2 Domain-Rule Imputation: Tạo Discount & Profit có quy luật kinh doanh
    if 'discount' not in df.columns or df['discount'].isna().all():
        cat_discount_map = {'Technology': 0.05, 'Furniture': 0.15, 'Office Supplies': 0.10}
        base_disc = df['category'].map(cat_discount_map).fillna(0.08) if 'category' in df.columns else 0.08
        bonus_disc = np.where(df['sales'] > 500, 0.05, 0.0) if 'sales' in df.columns else 0.0
        df['discount'] = np.clip(base_disc + bonus_disc, 0.0, 0.40).round(2)

    if 'profit' not in df.columns or df['profit'].isna().all():
        base_margin_map = {'Technology': 0.35, 'Furniture': 0.20, 'Office Supplies': 0.25}
        margin = df['category'].map(base_margin_map).fillna(0.25) if 'category' in df.columns else 0.25
        effective_margin = margin - (df['discount'] * 1.5)
        df['profit'] = (df['sales'] * effective_margin).round(2)

    # 3.3 Derived Metric
    if 'sales' in df.columns and 'profit' in df.columns:
        df['profit_margin'] = np.where(df['sales'] > 0, (df['profit'] / df['sales']) * 100, 0).round(2)

    # 3.4 Data Reduction: Loại bỏ các cột định danh/vị trí không phục vụ mô hình
    cols_to_drop = ['postal_code', 'country']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    return df


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def clean_and_transform_data(file_path: str) -> pd.DataFrame:
    """Hàm wrapper liên kết các module tiền xử lý theo luồng chuẩn."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding_errors='ignore')
    else:
        df = pd.read_excel(file_path)

    df_quality = process_data_quality(df)
    df_clean = handle_outliers_noise_consistency(df_quality)
    df_transformed = transform_and_reduce(df_clean)
    return df_transformed


def run_pipeline(file_path: str):
    """Chạy toàn bộ Pipeline: Clean -> Lưu vào SQLite."""
    print(f"🔄 Đang xử lý file: {file_path}...")
    df_clean = clean_and_transform_data(file_path)

    conn = create_connection()
    df_clean.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Đã xử lý xong {len(df_clean)} dòng dữ liệu và lưu vào SQLite ({DB_PATH})!")
    return df_clean


if __name__ == "__main__":
    sample_file = os.path.join("data", "train.csv")
    if os.path.exists(sample_file):
        run_pipeline(sample_file)
    else:
        print("[NOTICE] Chua co file mau trong data/. Hay tha 1 file Excel/CSV vao thu muc data/ de test.")