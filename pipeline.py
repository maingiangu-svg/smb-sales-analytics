import sqlite3
import re
import pandas as pd

DB_NAME = "data.db"


def find_header_row(file_path, sheet_name, max_rows=20):
    """Tìm dòng chứa header thật (dựa vào heuristic >= 2 cell không trống)."""
    preview = pd.read_excel(
        file_path, sheet_name=sheet_name, header=None, nrows=max_rows
    )
    for index, row in preview.iterrows():
        non_empty = row.dropna().astype(str).str.strip()
        if len(non_empty) >= 2:
            return index
    return 0


def normalize_headers(columns):
    """Đưa tên cột về dạng snake_case chuẩn SQL."""
    return (
        pd.Index(columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def clean_dataframe(df):
    """Xử lý rác dữ liệu cơ bản."""
    # Xóa dòng trắng toàn bộ
    df = df.dropna(how="all").copy()

    # Chuẩn hóa cột ngày tháng nếu có
    for col in df.columns:
        if "date" in col or "ngay" in col or "time" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def process_excel_to_sqlite(excel_file_path):
    """Đọc file Excel multi-tab và lưu trực tiếp vào SQLite."""
    excel = pd.ExcelFile(excel_file_path)
    conn = sqlite3.connect(DB_NAME)

    for sheet in excel.sheet_names:
        header_row = find_header_row(excel_file_path, sheet)
        df = pd.read_excel(excel_file_path, sheet_name=sheet, header=header_row)

        df.columns = normalize_headers(df.columns)
        df = clean_dataframe(df)

        # Đổi tên sheet thành tên bảng chuẩn SQL
        table_name = re.sub(r"[^a-zA-Z0-9]+", "_", sheet).lower().strip("_")

        # Ghi vào SQLite
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"✅ Loaded table '{table_name}': {len(df)} rows")

    conn.close()
    print("🚀 Dynamic Ingestion finished!")


if __name__ == "__main__":
    # Thay tên file Excel thô của bạn vào đây
    process_excel_to_sqlite("sample_data.xlsx")