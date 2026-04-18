"""
예측 성능 검증 유틸리티
"""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = [(float(a), float(p)) for a, p in zip(actual, predicted)]
    if not pairs:
        raise ValueError("RMSE를 계산할 데이터가 없습니다.")
    mse = sum((a - p) ** 2 for a, p in pairs) / len(pairs)
    return math.sqrt(mse)


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = [(float(a), float(p)) for a, p in zip(actual, predicted)]
    if not pairs:
        raise ValueError("MAE를 계산할 데이터가 없습니다.")
    return sum(abs(a - p) for a, p in pairs) / len(pairs)


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    valid_pairs = [(float(a), float(p)) for a, p in zip(actual, predicted) if float(a) != 0]
    if not valid_pairs:
        raise ValueError("MAPE를 계산할 0이 아닌 실제값이 없습니다.")
    return sum(abs((a - p) / a) for a, p in valid_pairs) / len(valid_pairs) * 100


def summarize_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    return {
        "RMSE": round(rmse(actual, predicted), 2),
        "MAE": round(mae(actual, predicted), 2),
        "MAPE(%)": round(mape(actual, predicted), 2),
    }


def seasonal_naive_forecast(
    series: pd.DataFrame,
    *,
    holdout_months: int,
    season_length: int = 12,
    value_col: str = "y",
) -> pd.DataFrame:
    """
    작년 같은 달 값을 그대로 가져오는 간단한 기준선 모델
    """
    if holdout_months <= 0:
        raise ValueError("holdout_months는 1 이상이어야 합니다.")

    ordered = series.sort_values("ds").reset_index(drop=True).copy()
    if len(ordered) <= holdout_months:
        raise ValueError("holdout 구간이 전체 길이보다 크거나 같습니다.")

    holdout = ordered.tail(holdout_months).copy()
    predictions = []
    for idx in holdout.index:
        reference_idx = idx - season_length
        if reference_idx < 0:
            reference_idx = idx - 1
        predictions.append(float(ordered.loc[reference_idx, value_col]))

    holdout["yhat"] = predictions
    return holdout[["ds", value_col, "yhat"]].rename(columns={value_col: "y"})


def build_comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        row = {"모델": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAPE(%)").reset_index(drop=True)
