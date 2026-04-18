"""
Prophet 기반 라면류 수요 예측

실행 예시:
    source venv/bin/activate
    python DemandForecast/prophet_model.py

기본 전략:
1. 2021-01-01 ~ 2023-06-01로 학습
2. 2023-07-01 ~ 2023-12-01로 holdout 검증
3. 그 뒤 전체 데이터(2021-01-01 ~ 2023-12-01)로 다시 학습
4. 2024-01-01 ~ 2026-12-01 미래 36개월 예측
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DemandForecast.forecast_queries import get_collection, monthly_series
from DemandForecast.validate import build_comparison_table, seasonal_naive_forecast, summarize_metrics

plt.rc("font", family="AppleGothic")
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_CATEGORY = "면류.라면류"
DEFAULT_HOLDOUT_MONTHS = 6
DEFAULT_FUTURE_MONTHS = 36
DEFAULT_OUTPUT_DIR = Path("DemandForecast/output")
DEFAULT_CONFIG = {
    "changepoint_prior_scale": 0.1,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "additive",
}
TUNING_GRID = [
    {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 0.1, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.1, "seasonality_prior_scale": 0.1, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.3, "seasonality_prior_scale": 0.1, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.3, "seasonality_prior_scale": 0.1, "seasonality_mode": "multiplicative"},
    {"changepoint_prior_scale": 0.5, "seasonality_prior_scale": 0.1, "seasonality_mode": "additive"},
]


def load_prophet_series(category: str = DEFAULT_CATEGORY) -> pd.DataFrame:
    col = get_collection()
    df = monthly_series(col, category=category).copy()
    if df.empty:
        raise ValueError(f"카테고리 데이터가 없습니다: {category}")

    result = df.rename(columns={"날짜": "ds", "판매수량": "y"})[["ds", "y", "거래건수"]].copy()
    result["ds"] = pd.to_datetime(result["ds"])
    result["y"] = result["y"].astype(float)
    result["거래건수"] = result["거래건수"].astype(float)
    return result.sort_values("ds").reset_index(drop=True)


def split_train_holdout(series: pd.DataFrame, holdout_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_months <= 0:
        raise ValueError("holdout_months는 1 이상이어야 합니다.")
    if len(series) <= holdout_months:
        raise ValueError("holdout_months가 전체 데이터 길이보다 크거나 같습니다.")

    train = series.iloc[:-holdout_months].copy()
    holdout = series.iloc[-holdout_months:].copy()
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def build_model(
    *,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
    seasonality_mode: str,
) -> Prophet:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
    )
    return model


def fit_and_predict(train_df: pd.DataFrame, periods: int, config: dict[str, float | str]) -> tuple[Prophet, pd.DataFrame]:
    model = build_model(**config)
    model.fit(train_df[["ds", "y"]])
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return model, forecast


def tune_prophet(series: pd.DataFrame, holdout_months: int) -> tuple[dict[str, float | str], pd.DataFrame]:
    train_df, holdout_df = split_train_holdout(series, holdout_months)
    tuning_rows = []
    best_config = None
    best_metrics = None

    for config in TUNING_GRID:
        _, forecast = fit_and_predict(train_df, periods=holdout_months, config=config)
        predicted = forecast[["ds", "yhat"]].copy()
        comparison = holdout_df.merge(predicted, on="ds", how="left")
        metrics = summarize_metrics(comparison["y"], comparison["yhat"])
        row = {
            "changepoint_prior_scale": config["changepoint_prior_scale"],
            "seasonality_prior_scale": config["seasonality_prior_scale"],
            "seasonality_mode": config["seasonality_mode"],
            **metrics,
        }
        tuning_rows.append(row)

        if best_metrics is None or metrics["MAPE(%)"] < best_metrics["MAPE(%)"]:
            best_config = config
            best_metrics = metrics

    tuning_df = pd.DataFrame(tuning_rows).sort_values("MAPE(%)").reset_index(drop=True)
    if best_config is None:
        raise ValueError("Prophet 튜닝에 실패했습니다.")
    return best_config, tuning_df


def evaluate_holdout(
    series: pd.DataFrame,
    holdout_months: int,
    tuned_config: dict[str, float | str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, holdout_df = split_train_holdout(series, holdout_months)

    _, default_forecast = fit_and_predict(train_df, periods=holdout_months, config=DEFAULT_CONFIG)
    _, tuned_forecast = fit_and_predict(train_df, periods=holdout_months, config=tuned_config)

    comparison = holdout_df.copy()
    comparison = comparison.merge(
        default_forecast[["ds", "yhat"]].rename(columns={"yhat": "기본Prophet"}),
        on="ds",
        how="left",
    )
    comparison = comparison.merge(
        tuned_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"yhat": "튜닝Prophet"}),
        on="ds",
        how="left",
    )
    comparison["튜닝Prophet"] = comparison["튜닝Prophet"].round(2)
    comparison["기본Prophet"] = comparison["기본Prophet"].round(2)
    comparison["yhat_lower"] = comparison["yhat_lower"].round(2)
    comparison["yhat_upper"] = comparison["yhat_upper"].round(2)

    default_metrics = summarize_metrics(comparison["y"], comparison["기본Prophet"])
    tuned_metrics = summarize_metrics(comparison["y"], comparison["튜닝Prophet"])
    naive_df = seasonal_naive_forecast(series, holdout_months=holdout_months)
    naive_metrics = summarize_metrics(naive_df["y"], naive_df["yhat"])

    metrics_table = build_comparison_table({
        "튜닝 Prophet": tuned_metrics,
        "기본 Prophet": default_metrics,
        "전년동월 기준선": naive_metrics,
    })
    return comparison, metrics_table


def forecast_future(
    series: pd.DataFrame,
    future_months: int,
    tuned_config: dict[str, float | str],
) -> tuple[Prophet, pd.DataFrame]:
    return fit_and_predict(series, periods=future_months, config=tuned_config)


def save_forecast_plot(
    series: pd.DataFrame,
    forecast: pd.DataFrame,
    holdout_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series["ds"], series["y"], color="#2f5d50", marker="o", label="실제값")
    ax.plot(forecast["ds"], forecast["yhat"], color="#c43d5c", linewidth=2, label="Prophet 예측")
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="#f4b6c2",
        alpha=0.3,
        label="신뢰구간",
    )
    ax.axvspan(
        holdout_df["ds"].min(),
        holdout_df["ds"].max(),
        color="#d9edf7",
        alpha=0.25,
        label="Holdout 구간",
    )
    ax.set_title("라면류 월별 판매수량: 실제값 vs Prophet 예측")
    ax.set_xlabel("날짜")
    ax.set_ylabel("판매수량")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_components_plot(model: Prophet, forecast: pd.DataFrame, output_path: Path) -> None:
    fig = model.plot_components(forecast)
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary(
    output_path: Path,
    *,
    category: str,
    holdout_months: int,
    series: pd.DataFrame,
    metrics_table: pd.DataFrame,
    tuned_config: dict[str, float | str],
) -> None:
    train_df, holdout_df = split_train_holdout(series, holdout_months)
    lines = [
        "라면류 Prophet 예측 요약",
        f"- 카테고리: {category}",
        f"- 학습 기간: {train_df['ds'].min().date()} ~ {train_df['ds'].max().date()}",
        f"- 검증 기간: {holdout_df['ds'].min().date()} ~ {holdout_df['ds'].max().date()}",
        f"- 전체 관측치 수: {len(series)}개월",
        f"- 선택된 Prophet 설정: cps={tuned_config['changepoint_prior_scale']}, "
        f"sps={tuned_config['seasonality_prior_scale']}, mode={tuned_config['seasonality_mode']}",
        "",
        metrics_table.to_string(index=False),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="라면류 Prophet 수요 예측")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="예측할 대분류")
    parser.add_argument("--holdout-months", type=int, default=DEFAULT_HOLDOUT_MONTHS, help="검증용 마지막 개월 수")
    parser.add_argument("--future-months", type=int, default=DEFAULT_FUTURE_MONTHS, help="미래 예측 개월 수")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="결과 저장 폴더")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series = load_prophet_series(category=args.category)
    tuned_config, tuning_df = tune_prophet(series, holdout_months=args.holdout_months)
    holdout_comparison, metrics_table = evaluate_holdout(
        series,
        holdout_months=args.holdout_months,
        tuned_config=tuned_config,
    )
    final_model, final_forecast = forecast_future(
        series,
        future_months=args.future_months,
        tuned_config=tuned_config,
    )

    forecast_csv = output_dir / "prophet_forecast.csv"
    holdout_csv = output_dir / "prophet_holdout_comparison.csv"
    metrics_csv = output_dir / "prophet_metrics.csv"
    tuning_csv = output_dir / "prophet_tuning_results.csv"
    summary_txt = output_dir / "prophet_summary.txt"
    forecast_png = output_dir / "prophet_forecast.png"
    components_png = output_dir / "prophet_components.png"

    final_forecast.to_csv(forecast_csv, index=False, encoding="utf-8-sig")
    holdout_comparison.to_csv(holdout_csv, index=False, encoding="utf-8-sig")
    metrics_table.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    tuning_df.to_csv(tuning_csv, index=False, encoding="utf-8-sig")
    write_summary(
        summary_txt,
        category=args.category,
        holdout_months=args.holdout_months,
        series=series,
        metrics_table=metrics_table,
        tuned_config=tuned_config,
    )
    save_forecast_plot(series, final_forecast, holdout_comparison, forecast_png)
    save_components_plot(final_model, final_forecast, components_png)

    print("=== 선택된 Prophet 설정 ===")
    print(tuned_config)

    print("\n=== Prophet 튜닝 결과 ===")
    print(tuning_df.to_string(index=False))

    print("=== Holdout 성능 비교 ===")
    print(metrics_table.to_string(index=False))

    print("\n=== Holdout 실제값 vs 예측값 ===")
    print(
        holdout_comparison[["ds", "y", "기본Prophet", "튜닝Prophet", "yhat_lower", "yhat_upper"]]
        .rename(columns={"ds": "날짜", "y": "실제값"})
        .to_string(index=False)
    )

    print("\n=== 저장 완료 ===")
    for path in [forecast_csv, holdout_csv, metrics_csv, tuning_csv, summary_txt, forecast_png, components_png]:
        print(path)


if __name__ == "__main__":
    main()
