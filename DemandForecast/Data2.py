"""
라면류 데이터 탐색 및 기초 분석 (MongoDB 버전)
실행 전: migrate_to_mongo.py 먼저 실행
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "retail_festa"
COLLECTION = "retail_sales"

plt.rc("font", family="AppleGothic")
plt.rcParams["axes.unicode_minus"] = False


def get_collection():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION]


def ramen_subcategory_share(col, year: int = 2023):
    """소분류별 판매 비중"""
    pipeline = [
        {"$match": {"대분류": "면류.라면류", "연도": year}},
        {"$group": {"_id": "$소분류", "총판매수량": {"$sum": "$판매수량"}}},
        {"$sort": {"총판매수량": -1}},
    ]
    results = list(col.aggregate(pipeline))
    labels = [r["_id"] for r in results]
    values = [r["총판매수량"] for r in results]

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title(f"{year}년 라면류 소분류별 판매 비중")
    plt.tight_layout()
    plt.show()


def monthly_sales_trend(col, year: int = 2023):
    """월별 판매수량 트렌드"""
    pipeline = [
        {"$match": {"대분류": "면류.라면류", "연도": year}},
        {"$group": {
            "_id": {"$month": "$판매일"},
            "총판매수량": {"$sum": "$판매수량"},
        }},
        {"$sort": {"_id": 1}},
    ]
    results = list(col.aggregate(pipeline))
    months = [r["_id"] for r in results]
    quantities = [r["총판매수량"] for r in results]

    plt.figure(figsize=(10, 5))
    plt.plot(months, quantities, marker="o")
    plt.xticks(range(1, 13))
    plt.xlabel("월")
    plt.ylabel("판매수량")
    plt.title(f"{year}년 라면류 월별 판매수량")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def top_products(col, year: int = 2023, top_n: int = 10):
    """상위 판매 상품"""
    pipeline = [
        {"$match": {"대분류": "면류.라면류", "연도": year}},
        {"$group": {"_id": "$상품명", "총판매수량": {"$sum": "$판매수량"}}},
        {"$sort": {"총판매수량": -1}},
        {"$limit": top_n},
    ]
    results = list(col.aggregate(pipeline))
    for i, r in enumerate(results, 1):
        print(f"{i:2}. {r['_id']:<30} {r['총판매수량']:>10,}개")


def null_summary(col, year: int = 2023):
    """결측 필드 현황"""
    total = col.count_documents({"대분류": "면류.라면류", "연도": year})
    for field in ["소분류", "규격", "옵션코드"]:
        null_count = col.count_documents({
            "대분류": "면류.라면류",
            "연도": year,
            field: {"$in": [None, ""]},
        })
        print(f"{field}: {null_count:,} / {total:,} ({null_count/total*100:.1f}%)")


if __name__ == "__main__":
    col = get_collection()

    print("=== 2023년 라면류 상위 10개 상품 ===")
    top_products(col, year=2023)

    print("\n=== 결측 현황 ===")
    null_summary(col, year=2023)

    monthly_sales_trend(col, year=2023)
    ramen_subcategory_share(col, year=2023)
