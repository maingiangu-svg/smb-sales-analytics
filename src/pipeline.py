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

    # 1.2 Map từ khóa cột (hỗ trợ cả Superstore lẫn Cafe dataset)
    col_map = {
        'total_spent': 'sales',
        'item': 'category',
        'transaction_date': 'order_date',
        'transaction_id': 'order_id'
    }
    for col in df.columns:
        if 'profit' in col and 'margin' not in col:
            col_map[col] = 'profit'
        elif 'discount' in col or 'disc' in col:
            col_map[col] = 'discount'
        elif ('sales' in col or 'revenue' in col) and col not in col_map:
            col_map[col] = 'sales'

    df.rename(columns=col_map, inplace=True)

    # 1.3 Khử trùng lặp (Duplicates)
    df = df.drop_duplicates()

    # 1.4 Làm sạch ký tự lạ, quy đổi 'ERROR'/'UNKNOWN' thành NaN và ép kiểu số
    # Thêm 'cost' vào danh sách cột số cần làm sạch
    numeric_cols = ['sales', 'quantity', 'price_per_unit', 'cost', 'discount', 'profit']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = (
                    df[col].astype(str)
                    .replace(['ERROR', 'UNKNOWN', 'unknown', 'None', 'nan'], np.nan)
                    .str.replace('$', '', regex=False)
                    .str.replace('%', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .str.strip()
                )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1.5 Khôi phục giá trị thiếu bằng toán học (Quantity * Price = Sales)
    if 'sales' in df.columns and 'quantity' in df.columns and 'price_per_unit' in df.columns:
        mask_sales = df['sales'].isna() & df['quantity'].notna() & df['price_per_unit'].notna()
        df.loc[mask_sales, 'sales'] = df['quantity'] * df['price_per_unit']

    # 1.6 Xử lý Missing Values
    if 'order_id' in df.columns:
        df = df.dropna(subset=['order_id'])
    
    cat_cols = ['category', 'sub_category', 'segment', 'region', 'payment_method', 'location']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].replace(['ERROR', 'UNKNOWN', 'nan'], np.nan).fillna("Unknown")

    return df


# ==============================================================================
# MODULE 2: OUTLIERS, NOISE & CONSISTENCY (Bài 4)
# ==============================================================================
def handle_outliers_noise_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tính nhất quán văn bản, lọc nhiễu logic và giới hạn ngoại lai."""
    df = df.copy()

    # 2.1 Consistency: Đồng nhất định dạng Title Case
    text_cols = ['category', 'sub_category', 'segment', 'region', 'payment_method', 'location']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # 2.2 Noise Filtering: Lọc bỏ đơn hàng có giá trị vô lý hoặc sai lệch ngày tháng
    if 'sales' in df.columns:
        df = df[df['sales'] > 0]

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])

    if 'order_date' in df.columns and 'ship_date' in df.columns:
        df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
        valid_dates = (df['ship_date'].isna()) | (df['ship_date'] >= df['order_date'])
        df = df[valid_dates]

    # 2.3 Outliers Handling: Áp dụng IQR Capping cho 'sales'
    if 'sales' in df.columns and len(df) > 0:
        q1 = df['sales'].quantile(0.25)
        q3 = df['sales'].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 3.0 * iqr
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

    # 3.2 Domain-Rule Imputation: Tạo Discount & Profit có quy luật
    if 'discount' not in df.columns or df['discount'].isna().all():
        np.random.seed(42)  # Cố định seed để dữ liệu nhất quán
        cat_discount_map = {
            'Coffee': 0.05, 'Tea': 0.05, 'Smoothie': 0.10, 'Juice': 0.10, 
            'Cake': 0.15, 'Sandwich': 0.12, 'Salad': 0.10, 'Cookie': 0.05,
            'Technology': 0.05, 'Furniture': 0.15, 'Office Supplies': 0.10
        }
        base_disc = df['category'].map(cat_discount_map).fillna(0.08) if 'category' in df.columns else 0.08
        bonus_disc = np.where(df['sales'] > 15, 0.05, 0.0) if 'sales' in df.columns else 0.0
        calculated_disc = np.clip(base_disc + bonus_disc, 0.0, 0.35).round(2)

        # Tạo nhóm đối chứng: ~30% số đơn giữ mức 0% chiết khấu để phục vụ T-test & ANOVA
        no_discount_mask = np.random.rand(len(df)) < 0.30
        df['discount'] = np.where(no_discount_mask, 0.0, calculated_disc)

    if 'profit' not in df.columns or df['profit'].isna().all():
        base_margin_map = {
            'Coffee': 0.70, 'Tea': 0.65, 'Smoothie': 0.60, 'Juice': 0.60, 
            'Cake': 0.55, 'Sandwich': 0.50, 'Salad': 0.50, 'Cookie': 0.60,
            'Technology': 0.35, 'Furniture': 0.20, 'Office Supplies': 0.25
        }
        margin = df['category'].map(base_margin_map).fillna(0.50) if 'category' in df.columns else 0.50
        effective_margin = margin - (df['discount'] * 1.5)
        df['profit'] = (df['sales'] * effective_margin).round(2)

    # 3.3 Derived Metric
    if 'sales' in df.columns and 'profit' in df.columns:
        df['profit_margin'] = np.where(df['sales'] > 0, (df['profit'] / df['sales']) * 100, 0).round(2)

    # 3.4 Data Reduction: Bỏ các cột dư thừa
    cols_to_drop = ['postal_code', 'country']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    return df


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
def clean_and_transform_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding_errors='ignore')
    else:
        df = pd.read_excel(file_path)

    df_quality = process_data_quality(df)
    df_clean = handle_outliers_noise_consistency(df_quality)
    df_transformed = transform_and_reduce(df_clean)
    return df_transformed


def run_pipeline(file_path: str):
    print(f"[PROCESS] Dang xu ly file: {file_path}...")
    df_clean = clean_and_transform_data(file_path)

    conn = create_connection()
    df_clean.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    output_csv = os.path.join("data", "cleaned_data.csv")
    df_clean.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"[OK] Da xu ly xong {len(df_clean)} dong du lieu!")
    print(f"[DATABASE] Da luu vao SQLite ({DB_PATH})")
    print(f"[EXPORT] Da xuat file CSV sach tai: {output_csv}")
    return df_clean


if __name__ == "__main__":
    # Ưu tiên dirty_cafe_sales.csv nếu có trong thư mục data/
    cafe_file = os.path.join("data", "dirty_cafe_sales.csv")
    train_file = os.path.join("data", "train.csv")
    
    target_file = cafe_file if os.path.exists(cafe_file) else train_file
    
    if os.path.exists(target_file):
        run_pipeline(target_file)
    else:
        print(f"[NOTICE] Chưa tìm thấy file dữ liệu tại data/ ({target_file})")