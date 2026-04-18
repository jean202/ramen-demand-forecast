"""
Excel(data2.xlsx) → MongoDB 마이그레이션 스크립트
실행: python migrate_to_mongo.py
사전 조건: pip install pymongo pandas openpyxl
"""
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import BulkWriteError

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "retail_festa"
COLLECTION_NAME = "retail_sales"
FILE_PATH = "data2.xlsx"
SHEETS = ["2021", "2022", "2023"]

# 소분류 결측치를 연도별 최빈값으로 채움
FILL_COLS = ["소분류", "규격"]


def load_sheet(year: str) -> pd.DataFrame:
    df = pd.read_excel(FILE_PATH, sheet_name=year)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    df["연도"] = int(year)
    for col in FILL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    if "판매일" in df.columns:
        df["판매일"] = pd.to_datetime(df["판매일"], errors="coerce")
    return df


def build_indexes(col):
    col.create_index([("판매일", ASCENDING)])
    col.create_index([("대분류", ASCENDING), ("판매일", ASCENDING)])
    col.create_index([("연도", ASCENDING), ("대분류", ASCENDING)])
    col.create_index([("매출처코드", ASCENDING)])
    col.create_index([("상품명", ASCENDING)])
    print("인덱스 생성 완료")


def migrate():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]

    col.drop()
    print(f"기존 컬렉션 초기화 완료: {COLLECTION_NAME}")

    total = 0
    for year in SHEETS:
        print(f"\n[{year}년] 데이터 로딩 중...")
        df = load_sheet(year)
        records = df.to_dict(orient="records")
        try:
            result = col.insert_many(records, ordered=False)
            inserted = len(result.inserted_ids)
        except BulkWriteError as e:
            inserted = e.details.get("nInserted", 0)
            print(f"  일부 오류 발생: {e.details['writeErrors'][:2]}")
        print(f"  {inserted:,}건 삽입 완료 (전체 {len(records):,}건)")
        total += inserted

    build_indexes(col)
    print(f"\n마이그레이션 완료: 총 {total:,}건 → {DB_NAME}.{COLLECTION_NAME}")
    client.close()


if __name__ == "__main__":
    migrate()
