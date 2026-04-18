"""
FIS PDF -> MongoDB 적재 스크립트

목표:
1. PDF 표를 추출한다.
2. 월별 시계열 형태로 길게 펼친다.
3. MongoDB retail_festa.fis_ramen_sales 컬렉션에 저장한다.

아직 PDF 샘플이 없어서, 표 구조가 달라도 최대한 버틸 수 있게
"긴 표(long format)"와 "월이 가로로 펼쳐진 표(wide format)"를 둘 다 시도한다.

권장 실행:
    source venv/bin/activate
    python DemandForecast/fis_migrate.py --pdf /absolute/path/to/fis.pdf --dry-run

실제 적재:
    python DemandForecast/fis_migrate.py --pdf /absolute/path/to/fis.pdf --replace-source

옵션 예시:
    python DemandForecast/fis_migrate.py \
        --pdf /absolute/path/to/fis.pdf \
        --parser camelot \
        --item-name 라면류 \
        --metric-label 매출액 \
        --save-csv DemandForecast/fis_ramen_sales_preview.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pymongo import ASCENDING, MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "retail_festa"
COLLECTION_NAME = "fis_ramen_sales"

METRIC_ALIASES = {
    "매출액": {"매출액", "매출", "금액", "매출금액", "sales", "value"},
    "판매량": {"판매량", "판매수량", "수량", "물량", "qty", "quantity"},
}

COLUMN_ALIASES = {
    "연도": {"연도", "년도", "year"},
    "월": {"월", "month", "month_num"},
    "품목": {"품목", "품명", "상품", "상품명", "분류", "구분", "category", "item"},
}

MONTH_PATTERN = re.compile(r"^(1[0-2]|0?[1-9])\s*월?$")
YEAR_PATTERN = re.compile(r"(20\d{2})")
YEAR_MONTH_PATTERN = re.compile(r"(20\d{2})\D{0,3}(1[0-2]|0?[1-9])\s*월?")
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(text: str) -> str:
    return normalize_text(text).lower().replace(" ", "").replace("_", "")


def parse_year(text) -> int | None:
    match = YEAR_PATTERN.search(normalize_text(text))
    return int(match.group(1)) if match else None


def parse_year_month(text) -> tuple[int, int] | None:
    match = YEAR_MONTH_PATTERN.search(normalize_text(text))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_month(text) -> int | None:
    cleaned = normalize_text(text)
    year_month = parse_year_month(cleaned)
    if year_month:
        return year_month[1]

    month_match = MONTH_PATTERN.match(cleaned.replace(".", "").replace("-", ""))
    if month_match:
        return int(month_match.group(1))

    if cleaned.isdigit():
        month = int(cleaned)
        if 1 <= month <= 12:
            return month
    return None


def parse_number(value) -> float | None:
    text = normalize_text(value)
    if not text or text in {"-", "nan", "None"}:
        return None
    match = NUMBER_PATTERN.search(text.replace("%", ""))
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def is_mostly_empty(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    cleaned = df.map(normalize_text)
    return cleaned.eq("").all().all()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [normalize_text(part) for part in col if normalize_text(part)]
            columns.append(" ".join(parts) if parts else "column")
        else:
            columns.append(normalize_text(col) or "column")
    result = df.copy()
    result.columns = columns
    return result


def cleanup_table(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df.copy())
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return df

    df = df.map(normalize_text)
    df = df.loc[~df.apply(lambda row: row.eq("").all(), axis=1)]
    df = df.loc[:, ~df.apply(lambda col: col.eq("").all(), axis=0)]
    return df.reset_index(drop=True)


def find_alias_column(columns: Iterable[str], alias_map: dict[str, set[str]], target: str) -> str | None:
    aliases = {normalize_key(alias) for alias in alias_map[target]}
    for column in columns:
        if normalize_key(column) in aliases:
            return column
    return None


def find_metric_columns(columns: Iterable[str]) -> list[tuple[str, str]]:
    results = []
    for column in columns:
        normalized = normalize_key(column)
        for metric_name, aliases in METRIC_ALIASES.items():
            if normalized in {normalize_key(alias) for alias in aliases}:
                results.append((column, metric_name))
                break
    return results


def build_standard_rows(
    frame: pd.DataFrame,
    *,
    year_col: str,
    month_col: str,
    value_col: str,
    metric_name: str,
    item_name: str,
    source_file: str,
    source_parser: str,
    table_index: int,
) -> pd.DataFrame:
    working = frame.copy()
    working["연도"] = working[year_col].apply(parse_year)
    working["월"] = working[month_col].apply(parse_month)
    working["값"] = working[value_col].apply(parse_number)

    item_col = find_alias_column(working.columns, COLUMN_ALIASES, "품목")
    if item_col:
        working["품목"] = working[item_col].replace("", item_name)
    else:
        working["품목"] = item_name

    result = working[["품목", "연도", "월", "값"]].copy()
    result["지표"] = metric_name
    result["날짜"] = pd.to_datetime(
        result["연도"].astype("Int64").astype(str) + "-"
        + result["월"].astype("Int64").astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )
    result["source_file"] = source_file
    result["source_parser"] = source_parser
    result["raw_table_index"] = table_index
    result = result.dropna(subset=["연도", "월", "값", "날짜"])
    result["연도"] = result["연도"].astype(int)
    result["월"] = result["월"].astype(int)
    return result[
        ["품목", "지표", "연도", "월", "날짜", "값", "source_file", "source_parser", "raw_table_index"]
    ].reset_index(drop=True)


def parse_long_format(
    df: pd.DataFrame,
    *,
    item_name: str,
    default_metric: str,
    source_file: str,
    source_parser: str,
    table_index: int,
) -> pd.DataFrame | None:
    year_col = find_alias_column(df.columns, COLUMN_ALIASES, "연도")
    month_col = find_alias_column(df.columns, COLUMN_ALIASES, "월")
    metric_cols = find_metric_columns(df.columns)

    if not year_col or not month_col:
        return None

    if not metric_cols:
        item_col = find_alias_column(df.columns, COLUMN_ALIASES, "품목")
        numeric_candidates = [
            col for col in df.columns
            if col not in {year_col, month_col, item_col}
            and df[col].apply(parse_number).notna().any()
        ]
        metric_cols = [(numeric_candidates[0], default_metric)] if numeric_candidates else []

    if not metric_cols:
        return None

    frames = [
        build_standard_rows(
            df,
            year_col=year_col,
            month_col=month_col,
            value_col=value_col,
            metric_name=metric_name,
            item_name=item_name,
            source_file=source_file,
            source_parser=source_parser,
            table_index=table_index,
        )
        for value_col, metric_name in metric_cols
    ]
    result = pd.concat(frames, ignore_index=True)
    return result if not result.empty else None


def parse_year_rows_format(
    df: pd.DataFrame,
    *,
    item_name: str,
    default_metric: str,
    source_file: str,
    source_parser: str,
    table_index: int,
) -> pd.DataFrame | None:
    month_columns = {column: parse_month(column) for column in df.columns}
    month_columns = {column: month for column, month in month_columns.items() if month is not None}
    if len(month_columns) < 3:
        return None

    non_month_columns = [column for column in df.columns if column not in month_columns]
    if not non_month_columns:
        return None

    year_col = None
    for candidate in non_month_columns:
        if df[candidate].apply(parse_year).notna().any():
            year_col = candidate
            break
    if not year_col:
        return None

    item_col = None
    for candidate in non_month_columns:
        if candidate == year_col:
            continue
        if normalize_key(candidate) in {normalize_key(alias) for alias in COLUMN_ALIASES["품목"]}:
            item_col = candidate
            break

    records = []
    for _, row in df.iterrows():
        year = parse_year(row[year_col])
        if year is None:
            continue
        item_value = normalize_text(row[item_col]) if item_col else item_name
        item_value = item_value or item_name
        for month_column, month in month_columns.items():
            value = parse_number(row[month_column])
            if value is None:
                continue
            records.append({
                "품목": item_value,
                "지표": default_metric,
                "연도": year,
                "월": month,
                "날짜": pd.Timestamp(year=year, month=month, day=1),
                "값": value,
                "source_file": source_file,
                "source_parser": source_parser,
                "raw_table_index": table_index,
            })

    if not records:
        return None
    return pd.DataFrame(records)


def parse_year_columns_format(
    df: pd.DataFrame,
    *,
    item_name: str,
    default_metric: str,
    source_file: str,
    source_parser: str,
    table_index: int,
) -> pd.DataFrame | None:
    year_columns = {column: parse_year(column) for column in df.columns}
    year_columns = {column: year for column, year in year_columns.items() if year is not None}
    if len(year_columns) < 2:
        return None

    non_year_columns = [column for column in df.columns if column not in year_columns]
    if not non_year_columns:
        return None

    month_col = None
    for candidate in non_year_columns:
        if df[candidate].apply(parse_month).notna().sum() >= min(3, len(df)):
            month_col = candidate
            break
    if not month_col:
        return None

    item_col = None
    for candidate in non_year_columns:
        if candidate == month_col:
            continue
        if normalize_key(candidate) in {normalize_key(alias) for alias in COLUMN_ALIASES["품목"]}:
            item_col = candidate
            break

    records = []
    for _, row in df.iterrows():
        month = parse_month(row[month_col])
        if month is None:
            continue
        item_value = normalize_text(row[item_col]) if item_col else item_name
        item_value = item_value or item_name
        for year_column, year in year_columns.items():
            value = parse_number(row[year_column])
            if value is None:
                continue
            records.append({
                "품목": item_value,
                "지표": default_metric,
                "연도": year,
                "월": month,
                "날짜": pd.Timestamp(year=year, month=month, day=1),
                "값": value,
                "source_file": source_file,
                "source_parser": source_parser,
                "raw_table_index": table_index,
            })

    if not records:
        return None
    return pd.DataFrame(records)


def parse_year_month_header_format(
    df: pd.DataFrame,
    *,
    item_name: str,
    default_metric: str,
    source_file: str,
    source_parser: str,
    table_index: int,
) -> pd.DataFrame | None:
    year_month_columns = {column: parse_year_month(column) for column in df.columns}
    year_month_columns = {column: value for column, value in year_month_columns.items() if value is not None}
    if len(year_month_columns) < 3:
        return None

    non_data_columns = [column for column in df.columns if column not in year_month_columns]
    item_col = None
    for candidate in non_data_columns:
        if normalize_key(candidate) in {normalize_key(alias) for alias in COLUMN_ALIASES["품목"]}:
            item_col = candidate
            break
    if item_col is None and non_data_columns:
        item_col = non_data_columns[0]

    records = []
    for _, row in df.iterrows():
        item_value = normalize_text(row[item_col]) if item_col else item_name
        item_value = item_value or item_name
        for column, (year, month) in year_month_columns.items():
            value = parse_number(row[column])
            if value is None:
                continue
            records.append({
                "품목": item_value,
                "지표": default_metric,
                "연도": year,
                "월": month,
                "날짜": pd.Timestamp(year=year, month=month, day=1),
                "값": value,
                "source_file": source_file,
                "source_parser": source_parser,
                "raw_table_index": table_index,
            })

    if not records:
        return None
    return pd.DataFrame(records)


def parse_tables(
    tables: list[pd.DataFrame],
    *,
    item_name: str,
    default_metric: str,
    source_file: str,
    source_parser: str,
) -> pd.DataFrame:
    parsed_frames = []
    failed_indices = []

    for index, raw_df in enumerate(tables):
        df = cleanup_table(raw_df)
        if is_mostly_empty(df):
            continue

        parsers = [
            parse_long_format,
            parse_year_rows_format,
            parse_year_columns_format,
            parse_year_month_header_format,
        ]

        parsed = None
        for parser in parsers:
            candidate = parser(
                df,
                item_name=item_name,
                default_metric=default_metric,
                source_file=source_file,
                source_parser=source_parser,
                table_index=index,
            )
            if candidate is not None and not candidate.empty:
                parsed = candidate
                break

        if parsed is None or parsed.empty:
            failed_indices.append(index)
            continue
        parsed_frames.append(parsed)

    if not parsed_frames:
        raise ValueError(
            "표는 읽었지만 월별 시계열로 자동 변환하지 못했습니다. "
            f"실패한 표 인덱스: {failed_indices or '없음'}. "
            "PDF를 받으면 실제 컬럼명을 보고 parse_* 함수 규칙을 맞추면 됩니다."
        )

    result = pd.concat(parsed_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["품목", "지표", "연도", "월", "값"])
    result = result.sort_values(["품목", "지표", "날짜"]).reset_index(drop=True)
    return result


def extract_with_camelot(pdf_path: Path, pages: str) -> list[pd.DataFrame]:
    try:
        import camelot  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "camelot-py가 설치되지 않았습니다. "
            "venv에서 `pip install camelot-py[cv]` 후 다시 실행하세요."
        ) from exc

    tables = camelot.read_pdf(str(pdf_path), pages=pages, flavor="stream")
    return [table.df for table in tables]


def extract_with_tabula(pdf_path: Path, pages: str) -> list[pd.DataFrame]:
    try:
        import tabula  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "tabula-py가 설치되지 않았습니다. "
            "venv에서 `pip install tabula-py` 후 다시 실행하세요."
        ) from exc

    dataframes = tabula.read_pdf(
        str(pdf_path),
        pages=pages,
        lattice=False,
        stream=True,
        multiple_tables=True,
        pandas_options={"dtype": str},
    )
    return [df for df in dataframes if isinstance(df, pd.DataFrame)]


def extract_tables(pdf_path: Path, parser_name: str, pages: str) -> tuple[list[pd.DataFrame], str]:
    errors = []

    if parser_name in {"auto", "camelot"}:
        try:
            return extract_with_camelot(pdf_path, pages), "camelot"
        except Exception as exc:
            errors.append(f"camelot 실패: {exc}")
            if parser_name == "camelot":
                raise

    if parser_name in {"auto", "tabula"}:
        try:
            return extract_with_tabula(pdf_path, pages), "tabula"
        except Exception as exc:
            errors.append(f"tabula 실패: {exc}")
            if parser_name == "tabula":
                raise

    raise RuntimeError(" / ".join(errors) if errors else "PDF 추출기 실행에 실패했습니다.")


def inspect_pdf_content(pdf_path: Path) -> dict[str, int]:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return {"pages": 0, "chars": 0, "images": 0, "words": 0}

    with pdfplumber.open(str(pdf_path)) as pdf:
        stats = {"pages": len(pdf.pages), "chars": 0, "images": 0, "words": 0}
        for page in pdf.pages:
            stats["chars"] += len(page.chars)
            stats["images"] += len(page.images)
            stats["words"] += len(page.extract_words())
    return stats


def validate_extracted_tables(pdf_path: Path, tables: list[pd.DataFrame]) -> None:
    if tables:
        return

    stats = inspect_pdf_content(pdf_path)
    if stats["pages"] and stats["chars"] == 0 and stats["images"] > 0:
        raise ValueError(
            "이 PDF는 표 텍스트가 아니라 이미지로 저장된 PDF입니다. "
            f"(pages={stats['pages']}, images={stats['images']}, chars={stats['chars']}) "
            "현재 파일만으로는 tabula/camelot 표 추출이 되지 않습니다. "
            "게다가 지금 받은 파일은 월별 표가 아니라 그래프 이미지라서 "
            "2021~2024 월별 시계열을 복원하기에 정보가 부족합니다. "
            "FIS에서 월별 숫자 표가 보이는 원본 PDF 또는 여러 장의 월별 캡처를 받아야 합니다."
        )

    raise ValueError(
        "PDF에서 읽힌 표가 없습니다. "
        "페이지 범위를 다시 지정하거나, 월별 숫자 표가 보이는 PDF인지 먼저 확인하세요."
    )


def build_indexes(col) -> None:
    col.create_index([("품목", ASCENDING), ("지표", ASCENDING), ("날짜", ASCENDING)])
    col.create_index([("source_file", ASCENDING)])
    col.create_index([("연도", ASCENDING), ("월", ASCENDING)])


def save_to_mongo(df: pd.DataFrame, *, source_file: str, replace_source: bool) -> int:
    client = MongoClient(MONGO_URI)
    try:
        col = client[DB_NAME][COLLECTION_NAME]

        if replace_source:
            deleted = col.delete_many({"source_file": source_file}).deleted_count
            print(f"[MongoDB] 기존 source_file={source_file} 문서 {deleted:,}건 삭제")

        records = df.to_dict(orient="records")
        if not records:
            return 0

        result = col.insert_many(records, ordered=False)
        build_indexes(col)
        return len(result.inserted_ids)
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FIS PDF 표를 월별 시계열로 바꿔 MongoDB에 적재합니다.")
    parser.add_argument("--pdf", required=True, help="FIS PDF 절대경로")
    parser.add_argument("--pages", default="all", help="읽을 페이지 범위. 예: all, 1, 2-4")
    parser.add_argument(
        "--parser",
        choices=["auto", "camelot", "tabula"],
        default="auto",
        help="PDF 표 추출기 선택",
    )
    parser.add_argument("--item-name", default="라면류", help="품목명이 PDF에 없을 때 넣을 기본값")
    parser.add_argument("--metric-label", default="매출액", help="지표명이 PDF에 없을 때 넣을 기본값")
    parser.add_argument("--save-csv", help="정리된 결과를 CSV로 저장할 경로")
    parser.add_argument("--dry-run", action="store_true", help="MongoDB에 넣지 않고 미리보기만 출력")
    parser.add_argument(
        "--replace-source",
        action="store_true",
        help="같은 PDF 파일명(source_file)으로 적재된 문서를 먼저 삭제",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    tables, actual_parser = extract_tables(pdf_path, args.parser, args.pages)
    print(f"[추출] parser={actual_parser}, tables={len(tables)}")
    validate_extracted_tables(pdf_path, tables)

    parsed_df = parse_tables(
        tables,
        item_name=args.item_name,
        default_metric=args.metric_label,
        source_file=pdf_path.name,
        source_parser=actual_parser,
    )

    print(f"[정리] rows={len(parsed_df):,}")
    print(parsed_df.head(12).to_string(index=False))

    if args.save_csv:
        csv_path = Path(args.save_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[저장] CSV 저장 완료: {csv_path}")

    if args.dry_run:
        print("[완료] dry-run 모드라 MongoDB 적재는 생략했습니다.")
        return

    inserted = save_to_mongo(
        parsed_df,
        source_file=pdf_path.name,
        replace_source=args.replace_source,
    )
    print(f"[완료] MongoDB 적재 {inserted:,}건 -> {DB_NAME}.{COLLECTION_NAME}")


if __name__ == "__main__":
    main()
