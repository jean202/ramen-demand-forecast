"""
수요 예측을 위한 MongoDB 집계 쿼리 모음
반환값은 모두 pandas DataFrame — 이후 sklearn/statsmodels 모델 입력으로 사용
"""
import pandas as pd
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "retail_festa"
COLLECTION = "retail_sales"


def get_collection():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION]


def monthly_series(col, category: str = "면류.라면류") -> pd.DataFrame:
    """
    연도-월별 판매수량 시계열 (3개년 전체)
    수요 예측 모델(Prophet, ARIMA 등)의 기본 입력 데이터
    """
    pipeline = [
        {"$match": {"대분류": category}},
        {"$group": {
            "_id": {
                "연도": "$연도",
                "월": {"$month": "$판매일"},
            },
            "판매수량": {"$sum": "$판매수량"},
            "거래건수": {"$sum": 1},
        }},
        {"$sort": {"_id.연도": 1, "_id.월": 1}},
    ]
    rows = list(col.aggregate(pipeline))
    df = pd.DataFrame([{
        "연도": r["_id"]["연도"],
        "월": r["_id"]["월"],
        "판매수량": r["판매수량"],
        "거래건수": r["거래건수"],
    } for r in rows])
    df["날짜"] = pd.to_datetime(df["연도"].astype(str) + "-" + df["월"].astype(str) + "-01")
    return df.sort_values("날짜").reset_index(drop=True)


def yoy_comparison(col, category: str = "면류.라면류") -> pd.DataFrame:
    """
    연도별 월 판매수량 피벗 (전년 대비 증감률 계산용)
    """
    df = monthly_series(col, category)
    pivot = df.pivot(index="월", columns="연도", values="판매수량").reset_index()
    for yr in [2022, 2023]:
        if yr in pivot.columns and yr - 1 in pivot.columns:
            pivot[f"{yr}_증감률(%)"] = ((pivot[yr] - pivot[yr - 1]) / pivot[yr - 1] * 100).round(2)
    return pivot


def subcategory_monthly(col, category: str = "면류.라면류") -> pd.DataFrame:
    """
    소분류 × 연도-월 판매수량 (다변량 예측용)
    """
    pipeline = [
        {"$match": {"대분류": category}},
        {"$group": {
            "_id": {
                "소분류": "$소분류",
                "연도": "$연도",
                "월": {"$month": "$판매일"},
            },
            "판매수량": {"$sum": "$판매수량"},
        }},
        {"$sort": {"_id.소분류": 1, "_id.연도": 1, "_id.월": 1}},
    ]
    rows = list(col.aggregate(pipeline))
    df = pd.DataFrame([{
        "소분류": r["_id"]["소분류"],
        "연도": r["_id"]["연도"],
        "월": r["_id"]["월"],
        "판매수량": r["판매수량"],
    } for r in rows])
    df["날짜"] = pd.to_datetime(df["연도"].astype(str) + "-" + df["월"].astype(str) + "-01")
    return df.sort_values(["소분류", "날짜"]).reset_index(drop=True)


def regional_monthly(col, category: str = "면류.라면류") -> pd.DataFrame:
    """
    우편번호 앞 3자리(지역) × 월별 판매수량 (지역별 수요 패턴 분석용)
    """
    pipeline = [
        {"$match": {"대분류": category, "우편번호": {"$ne": None}}},
        {"$addFields": {
            "지역코드": {"$substr": [{"$toString": "$우편번호"}, 0, 3]},
        }},
        {"$group": {
            "_id": {
                "지역코드": "$지역코드",
                "연도": "$연도",
                "월": {"$month": "$판매일"},
            },
            "판매수량": {"$sum": "$판매수량"},
        }},
        {"$sort": {"_id.지역코드": 1, "_id.연도": 1, "_id.월": 1}},
    ]
    rows = list(col.aggregate(pipeline))
    df = pd.DataFrame([{
        "지역코드": r["_id"]["지역코드"],
        "연도": r["_id"]["연도"],
        "월": r["_id"]["월"],
        "판매수량": r["판매수량"],
    } for r in rows])
    df["날짜"] = pd.to_datetime(df["연도"].astype(str) + "-" + df["월"].astype(str) + "-01")
    return df.sort_values(["지역코드", "날짜"]).reset_index(drop=True)


def top_products_trend(col, category: str = "면류.라면류", top_n: int = 5) -> pd.DataFrame:
    """
    2021~2023 누적 상위 N개 상품의 월별 시계열
    """
    top_pipeline = [
        {"$match": {"대분류": category}},
        {"$group": {"_id": "$상품명", "총판매수량": {"$sum": "$판매수량"}}},
        {"$sort": {"총판매수량": -1}},
        {"$limit": top_n},
    ]
    top_names = [r["_id"] for r in col.aggregate(top_pipeline)]

    pipeline = [
        {"$match": {"대분류": category, "상품명": {"$in": top_names}}},
        {"$group": {
            "_id": {
                "상품명": "$상품명",
                "연도": "$연도",
                "월": {"$month": "$판매일"},
            },
            "판매수량": {"$sum": "$판매수량"},
        }},
        {"$sort": {"_id.상품명": 1, "_id.연도": 1, "_id.월": 1}},
    ]
    rows = list(col.aggregate(pipeline))
    df = pd.DataFrame([{
        "상품명": r["_id"]["상품명"],
        "연도": r["_id"]["연도"],
        "월": r["_id"]["월"],
        "판매수량": r["판매수량"],
    } for r in rows])
    df["날짜"] = pd.to_datetime(df["연도"].astype(str) + "-" + df["월"].astype(str) + "-01")
    return df.sort_values(["상품명", "날짜"]).reset_index(drop=True)


if __name__ == "__main__":
    col = get_collection()

    print("=== 월별 시계열 (head) ===")
    print(monthly_series(col).head(12).to_string(index=False))

    print("\n=== YoY 비교 ===")
    print(yoy_comparison(col).to_string(index=False))

    print("\n=== 소분류별 월 판매 (head) ===")
    print(subcategory_monthly(col).head(10).to_string(index=False))

    print("\n=== 상위 5개 상품 트렌드 (head) ===")
    print(top_products_trend(col, top_n=5).head(15).to_string(index=False))
