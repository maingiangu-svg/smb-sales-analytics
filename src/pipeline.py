import os
import re
import sys
import sqlite3
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DB_PATH = os.path.join("data", "data.db")
TABLE_NAME = "sales_data"
DEFAULT_RAW_FILE = os.path.join("data", "dirty_cafe_sales.csv")
CLEAN_CSV_PATH = os.path.join("data", "cleaned_cafe_sales.csv")


def normalize_headers(columns):
    """Bước 2: Chuẩn hóa tên cột thành chữ thường, regex thay ký tự lạ bằng dấu gạch dưới."""
    return [re.sub(r'[^a-zA-Z0-9]+', '_', str(c).strip().lower()).strip('_') for c in columns]


def step2_normalize_headers_and_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 2: Chuẩn hóa Header
    - Chuyển tên cột thành chuỗi, xóa khoảng trắng thừa ở 2 đầu, đưa về lowercase.
    - Đưa mọi ký tự không phải chữ cái và số về '_' và cắt bỏ '_' thừa.
    - Đổi tên cột chuẩn (sales, category, order_date, order_id).
    - Loại bỏ các dòng trùng lặp hoàn toàn.
    """
    df = df.copy()
    df.columns = normalize_headers(df.columns)
    col_map = {
        'total_spent': 'sales',
        'item': 'category',
        'transaction_date': 'order_date',
        'transaction_id': 'order_id'
    }
    df.rename(columns=col_map, inplace=True)
    df = df.drop_duplicates()
    return df


def step3_create_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 3: Tạo indicator columns bảo tồn ngữ nghĩa
    - TẠO CHO PAYMENT METHOD (MCAR + MNAR): Đánh dấu 1 nếu là UNKNOWN (khách trả Bitcoin/chuyển khoản).
    - TẠO CHO LOCATION (Missing 39.61%): Đánh dấu 1 nếu là ERROR (lỗi kỹ thuật POS).
    """
    df = df.copy()
    if 'payment_method' in df.columns:
        df['is_unconventional_payment'] = np.where(
            df['payment_method'].astype(str).str.strip().str.upper() == 'UNKNOWN', 1, 0
        )
    else:
        df['is_unconventional_payment'] = 0

    if 'location' in df.columns:
        df['location_has_error'] = np.where(
            df['location'].astype(str).str.strip().str.upper() == 'ERROR', 1, 0
        )
    else:
        df['location_has_error'] = 0

    return df


def step4_clean_and_cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 4: Đồng bộ missing values về NaN / NaT và ép kiểu dữ liệu đúng
    1. Các cột numeric: sales, quantity, price_per_unit, cost:
       - Bỏ dấu $, %, dấu phẩy ,, cắt khoảng trắng thừa.
    2. Biến các chuỗi rác thành np.nan (TRỪ CÁC CỘT CỜ HIỆU ĐÃ TẠO).
    3. Ép kiểu dữ liệu:
       - Ép các cột số về float64 (chữ lạ tự chuyển thành NaN qua errors='coerce').
       - Ép cột ngày tháng order_date về datetime64 (ngày lỗi tự chuyển thành NaT).
    """
    df = df.copy()
    numeric_cols = ['sales', 'quantity', 'price_per_unit', 'cost']
    
    # 1. Làm sạch ký tự hiển thị ở cột số
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace('$', '', regex=False)
                .str.replace('%', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )

    # 2. Quy chuẩn hóa missing values về np.nan
    garbage_strings = ['ERROR', 'UNKNOWN', 'unknown', 'None', 'nan', '']
    excluded_cols = ['is_unconventional_payment', 'location_has_error']
    for col in df.columns:
        if col not in excluded_cols:
            df[col] = df[col].replace(garbage_strings, np.nan)

    # 3. Ép kiểu dữ liệu an toàn
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    for col in ['category', 'payment_method', 'location']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title().replace(['Nan', 'None', 'Unknown', ''], np.nan)

    return df


def step5_impute_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 5: Xử lý khuyết thiếu
    1. Xóa các dòng MCAR < 5% (Chỉ có transaction_date bị xóa, order_id nếu có).
    2. Điền đơn giá nếu biết tên món, tên món nếu biết đơn giá qua Menu Cafe.
    3. Khôi phục Sales = Quantity * Price Per Unit.
    4. Làm sạch số và điền Group Median cho Cost:
       - Khử nhiễu dấu âm (abs).
       - Fix outlier nhân 100 lần (/ 100.0).
       - Điền Group Median theo category vào các ô NaN.
    5. Điền chữ thay thế vào các ô NaN của payment_method ('Other_or_Crypto'), location ('Unknown'), item ('Unknown').
    """
    df = df.copy()

    # 1. Xóa dòng MCAR < 5%
    if 'order_date' in df.columns:
        df = df.dropna(subset=['order_date'])
    if 'order_id' in df.columns:
        df = df.dropna(subset=['order_id'])

    # 2. Menu Cafe: Khôi phục qua lại giữa Category và Price Per Unit
    menu_price = {
        'Cookie': 1.0, 'Tea': 1.5, 'Coffee': 2.0,
        'Cake': 3.0, 'Juice': 3.0, 'Sandwich': 4.0, 'Smoothie': 4.0, 'Salad': 5.0
    }
    if 'category' in df.columns and 'price_per_unit' in df.columns:
        df['price_per_unit'] = df['price_per_unit'].fillna(df['category'].map(menu_price))
        price_to_cat = {1.0: 'Cookie', 1.5: 'Tea', 2.0: 'Coffee', 5.0: 'Salad'}
        mask_cat = df['category'].isna() & df['price_per_unit'].isin(price_to_cat.keys())
        df.loc[mask_cat, 'category'] = df.loc[mask_cat, 'price_per_unit'].map(price_to_cat)

    # 3. Khôi phục toán học 3 chiều: Sales = Quantity * Price Per Unit
    if {'sales', 'quantity', 'price_per_unit'}.issubset(df.columns):
        m_sales = df['sales'].isna() & df['quantity'].notna() & df['price_per_unit'].notna()
        df.loc[m_sales, 'sales'] = df.loc[m_sales, 'quantity'] * df.loc[m_sales, 'price_per_unit']

        m_qty = df['quantity'].isna() & df['sales'].notna() & (df['price_per_unit'] > 0)
        df.loc[m_qty, 'quantity'] = (df.loc[m_qty, 'sales'] / df.loc[m_qty, 'price_per_unit']).round()

        m_price = df['price_per_unit'].isna() & df['sales'].notna() & (df['quantity'] > 0)
        df.loc[m_price, 'price_per_unit'] = (df.loc[m_price, 'sales'] / df.loc[m_price, 'quantity']).round(2)

        # Fallback cho các dòng khuyết cả 2 ẩn
        df['price_per_unit'] = df['price_per_unit'].fillna(df.groupby('category')['price_per_unit'].transform('median')).fillna(df['price_per_unit'].median())
        df['quantity'] = df['quantity'].fillna(df.groupby('category')['quantity'].transform('median')).fillna(df['quantity'].median())
        df['sales'] = df['sales'].fillna(df['quantity'] * df['price_per_unit'])

    # 4. Làm sạch số và điền Group Median cho Cost
    if 'cost' in df.columns:
        df['cost'] = df['cost'].abs()
        df['cost'] = np.where(df['cost'] >= 10, df['cost'] / 100.0, df['cost'])
        df.loc[df['cost'] <= 0, 'cost'] = np.nan
        df['cost'] = df['cost'].fillna(df.groupby('category')['cost'].transform('median')).fillna(df['cost'].median())

    # 5. Điền chữ thay thế cho các cột phân loại
    if 'payment_method' in df.columns:
        df['payment_method'] = df['payment_method'].fillna('Other_or_Crypto')
    if 'location' in df.columns:
        df['location'] = df['location'].fillna('Unknown')
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('Unknown')

    return df


def step6_enforce_business_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bước 6: Kiểm soát ràng buộc nghiệp vụ
    1. Xử lý nghiệp vụ cho Cost: Áp dụng bảng % Food Cost theo món chuẩn năm 2023:
       - Coffee: 18–25%
       - Smoothie: 28–36%
       - Juice: 28–36%
       - Cookie: 20–30%
       - Cake: 20–30%
       - Salad: 28–33%
       - Sandwich: 28–35%
       - Default: 20–30%
    2. Bóc tách đặc trưng thời gian (year, month, quarter, day_of_week) phục vụ mô hình dự báo doanh số theo tháng.
    """
    df = df.copy()

    cost_ratio_bounds = {
        'Coffee':   (0.18, 0.25),
        'Smoothie': (0.28, 0.36),
        'Juice':    (0.28, 0.36),
        'Cookie':   (0.20, 0.30),
        'Cake':     (0.20, 0.30),
        'Salad':    (0.28, 0.33),
        'Sandwich': (0.28, 0.35)
    }

    if {'cost', 'price_per_unit', 'category'}.issubset(df.columns):
        lower_ratios = df['category'].map(lambda c: cost_ratio_bounds.get(c, (0.20, 0.30))[0])
        upper_ratios = df['category'].map(lambda c: cost_ratio_bounds.get(c, (0.20, 0.30))[1])

        lower_cost = df['price_per_unit'] * lower_ratios
        upper_cost = df['price_per_unit'] * upper_ratios

        # Chặn biên chi phí chuẩn xác cho từng dòng
        df['cost'] = np.clip(df['cost'], lower_cost, upper_cost).round(2)

    # Trích xuất đặc trưng thời gian phục vụ dự báo doanh thu chuỗi thời gian
    if 'order_date' in df.columns:
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['quarter'] = df['order_date'].dt.quarter
        df['day_of_week'] = df['order_date'].dt.day_name()

    return df


def clean_and_transform_data(file_path: str = DEFAULT_RAW_FILE) -> pd.DataFrame:
    """Thực thi tuần tự toàn bộ Data Pipeline từ Bước 2 đến Bước 6."""
    df_raw = pd.read_csv(file_path, keep_default_na=False) if file_path.endswith('.csv') else pd.read_excel(file_path)
    
    df_s2 = step2_normalize_headers_and_schema(df_raw)
    df_s3 = step3_create_indicators(df_s2)
    df_s4 = step4_clean_and_cast_types(df_s3)
    df_s5 = step5_impute_missing_data(df_s4)
    df_s6 = step6_enforce_business_constraints(df_s5)
    
    return df_s6


def run_pipeline(file_path: str = None):
    """Chạy toàn bộ pipeline, lưu CSDL SQLite và xuất file CSV sạch sẵn sàng sử dụng."""
    if file_path is None:
        file_path = DEFAULT_RAW_FILE
        
    print(f"🔄 Đang thực thi Data Pipeline trên: {file_path}...")
    df_clean = clean_and_transform_data(file_path)

    os.makedirs("data", exist_ok=True)
    
    # 1. Lưu vào Database SQLite
    with sqlite3.connect(DB_PATH) as conn:
        df_clean.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    # 2. Xuất sẵn file CSV sạch ra thư mục data
    df_clean.to_csv(CLEAN_CSV_PATH, index=False, encoding='utf-8')

    print(f"✅ Đã xử lý xong {len(df_clean)} dòng sạch (0 ô NaN) và lưu vào {DB_PATH} & {CLEAN_CSV_PATH}!")
    return df_clean


if __name__ == "__main__":
    run_pipeline()
