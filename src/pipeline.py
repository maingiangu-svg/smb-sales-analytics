import os
import re
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = os.path.join("data", "data.db")
TABLE_NAME = "sales_data"


def normalize_headers(columns):
    return [re.sub(r'[^a-zA-Z0-9]+', '_', str(c).strip().lower()).strip('_') for c in columns]


def process_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Bài 3: Làm sạch Data Quality, khôi phục chéo Sales - Quantity - Price - Category."""
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

    # Làm sạch ký tự rác ở các cột số
    numeric_cols = ['sales', 'quantity', 'price_per_unit', 'cost', 'discount', 'profit']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .replace(['ERROR', 'UNKNOWN', 'unknown', 'None', 'nan'], np.nan)
                .str.replace('$', '', regex=False)
                .str.replace('%', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['category', 'payment_method', 'location']:
        if col in df.columns:
            df[col] = df[col].replace(['ERROR', 'UNKNOWN', 'nan', 'None'], np.nan)

    # Bảng giá chuẩn của Menu Cafe để khôi phục chéo
    menu_price = {
        'Cookie': 1.0, 'Tea': 1.5, 'Coffee': 2.0,
        'Cake': 3.0, 'Juice': 3.0, 'Sandwich': 4.0, 'Smoothie': 4.0, 'Salad': 5.0
    }
    if 'category' in df.columns and 'price_per_unit' in df.columns:
        df['price_per_unit'] = df['price_per_unit'].fillna(df['category'].map(menu_price))

    # Khôi phục toán học 3 chiều: Sales = Quantity * Price Per Unit
    if {'sales', 'quantity', 'price_per_unit'}.issubset(df.columns):
        m_sales = df['sales'].isna() & df['quantity'].notna() & df['price_per_unit'].notna()
        df.loc[m_sales, 'sales'] = df.loc[m_sales, 'quantity'] * df.loc[m_sales, 'price_per_unit']

        m_qty = df['quantity'].isna() & df['sales'].notna() & (df['price_per_unit'] > 0)
        df.loc[m_qty, 'quantity'] = (df.loc[m_qty, 'sales'] / df.loc[m_qty, 'price_per_unit']).round()

        m_price = df['price_per_unit'].isna() & df['sales'].notna() & (df['quantity'] > 0)
        df.loc[m_price, 'price_per_unit'] = (df.loc[m_price, 'sales'] / df.loc[m_price, 'quantity']).round(2)

        # Điền nốt các dòng khuyết cả 2 chiều bằng trung vị để không còn bất kỳ ô NaN nào
        df['price_per_unit'] = df['price_per_unit'].fillna(df['price_per_unit'].median())
        df['quantity'] = df['quantity'].fillna((df['sales'] / df['price_per_unit']).round()).fillna(df['quantity'].median())
        df['sales'] = df['sales'].fillna(df['quantity'] * df['price_per_unit'])

    # Suy ngược tên món ăn từ đơn giá đặc trưng
    price_to_cat = {1.0: 'Cookie', 1.5: 'Tea', 2.0: 'Coffee', 5.0: 'Salad'}
    if 'category' in df.columns and 'price_per_unit' in df.columns:
        m_cat = df['category'].isna() & df['price_per_unit'].isin(price_to_cat.keys())
        df.loc[m_cat, 'category'] = df.loc[m_cat, 'price_per_unit'].map(price_to_cat)

    if 'order_id' in df.columns:
        df = df.dropna(subset=['order_id'])

    for col in ['category', 'payment_method', 'location']:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


def handle_outliers_noise_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Bài 4: Xử lý nhiễu, chuẩn hóa văn bản và chặn ngoại lai (IQR)."""
    df = df.copy()
    for col in ['category', 'payment_method', 'location']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    if 'sales' in df.columns:
        df = df[df['sales'] > 0]

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])

    # Xử lý nhiễu âm và ngoại lai gấp 100 lần ở cột Cost (ví dụ -1.2 -> 1.2; 120.0 -> 1.2)
    if 'cost' in df.columns:
        df['cost'] = df['cost'].abs()
        df['cost'] = np.where(df['cost'] >= 10, df['cost'] / 100.0, df['cost'])
        df.loc[df['cost'] <= 0, 'cost'] = np.nan
        df['cost'] = df['cost'].fillna(df.groupby('category')['cost'].transform('median')).fillna(df['cost'].median())
    else:
        df['cost'] = 1.0

    # IQR Capping cho Sales
    if 'sales' in df.columns and len(df) > 0:
        q1, q3 = df['sales'].quantile(0.25), df['sales'].quantile(0.75)
        upper_limit = q3 + 3.0 * (q3 - q1)
        df['sales'] = np.where(df['sales'] > upper_limit, upper_limit, df['sales'])

    return df


def transform_and_reduce(df: pd.DataFrame) -> pd.DataFrame:
    """Bài 5: Trích xuất đặc trưng thời gian và tính toán Lợi nhuận từ Giá vốn thực."""
    df = df.copy()
    if 'order_date' in df.columns:
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day'] = df['order_date'].dt.day
        df['quarter'] = df['order_date'].dt.quarter
        df['day_of_week'] = df['order_date'].dt.day_name()

    if 'discount' not in df.columns or df['discount'].isna().all():
        np.random.seed(42)
        cat_discount_map = {
            'Coffee': 0.05, 'Tea': 0.05, 'Smoothie': 0.10, 'Juice': 0.10,
            'Cake': 0.15, 'Sandwich': 0.12, 'Salad': 0.10, 'Cookie': 0.05
        }
        base_disc = df['category'].map(cat_discount_map).fillna(0.08)
        bonus_disc = np.where(df['sales'] >= 12, 0.08, 0.0)
        promo_shock = np.random.choice([0.0, 0.05, 0.10, 0.15], size=len(df), p=[0.5, 0.25, 0.15, 0.10])
        calculated_disc = np.clip(base_disc + bonus_disc + promo_shock, 0.0, 0.35).round(2)

        no_discount_mask = np.random.rand(len(df)) < 0.30
        df['discount'] = np.where(no_discount_mask, 0.0, calculated_disc)

    if 'profit' not in df.columns or df['profit'].isna().all():
        base_margin_map = {
            'Coffee': 0.72, 'Tea': 0.68, 'Smoothie': 0.62, 'Juice': 0.62,
            'Cake': 0.58, 'Sandwich': 0.54, 'Salad': 0.55, 'Cookie': 0.65
        }
        margin = df['category'].map(base_margin_map).fillna(0.55)
        discount_penalty = df['discount'] * 1.3 + np.where(df['discount'] > 0.20, (df['discount'] - 0.20) * 1.8, 0.0)
        df['profit'] = (df['sales'] * (margin - discount_penalty) - df['cost'] * 0.85).round(2)

    df['profit_margin'] = np.where(df['sales'] > 0, (df['profit'] / df['sales']) * 100, 0).round(2)
    return df


def clean_and_transform_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding_errors='ignore') if file_path.endswith('.csv') else pd.read_excel(file_path)
    return transform_and_reduce(handle_outliers_noise_consistency(process_data_quality(df)))


def run_pipeline(file_path: str = None):
    if file_path is None:
        file_path = os.path.join("data", "dirty_cafe_sales.csv")
    print(f"🔄 Đang xử lý file: {file_path}...")
    df_clean = clean_and_transform_data(file_path)

    os.makedirs("data", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        df_clean.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print(f"✅ Đã xử lý xong {len(df_clean)} dòng sạch (0 NaN) và lưu vào {DB_PATH}!")
    return df_clean


if __name__ == "__main__":
    run_pipeline()