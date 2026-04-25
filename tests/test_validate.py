import math

import pandas as pd
import pytest

from DemandForecast.validate import (
    build_comparison_table,
    mae,
    mape,
    rmse,
    seasonal_naive_forecast,
    summarize_metrics,
)


class TestRmse:
    def test_perfect_prediction(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_basic(self):
        # errors [2, 2] → mse=4 → rmse=2
        assert math.isclose(rmse([3, 3], [1, 1]), 2.0)

    def test_asymmetric(self):
        # errors [1, 3] → mse=5 → rmse=√5
        assert math.isclose(rmse([1, 3], [0, 0]), math.sqrt(5))

    def test_empty_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            rmse([], [])


class TestMae:
    def test_perfect_prediction(self):
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_basic(self):
        # abs errors [2, 2] → mean=2
        assert math.isclose(mae([3, 3], [1, 1]), 2.0)

    def test_asymmetric(self):
        # abs errors [1, 3] → mean=2
        assert math.isclose(mae([1, 3], [0, 0]), 2.0)


class TestMape:
    def test_perfect_prediction(self):
        assert mape([100, 200], [100, 200]) == 0.0

    def test_ten_percent_error(self):
        # actual=100, predicted=90 → 10%
        assert math.isclose(mape([100], [90]), 10.0)

    def test_skips_zero_actual(self):
        # actual=0 건너뜀, actual=100/predicted=90 → 10%
        assert math.isclose(mape([0, 100], [999, 90]), 10.0)

    def test_all_zero_actual_raises(self):
        with pytest.raises(ValueError):
            mape([0, 0], [1, 2])


class TestSummarizeMetrics:
    def test_returns_required_keys(self):
        result = summarize_metrics([100], [90])
        assert set(result.keys()) == {"RMSE", "MAE", "MAPE(%)"}

    def test_values_are_rounded(self):
        result = summarize_metrics([100], [90])
        for v in result.values():
            assert isinstance(v, float)
            assert round(v, 2) == v


class TestSeasonalNaiveForecast:
    def _make_series(self, n: int) -> pd.DataFrame:
        ds = pd.date_range("2021-01-01", periods=n, freq="MS")
        return pd.DataFrame({"ds": ds, "y": range(1, n + 1)})

    def test_output_length(self):
        series = self._make_series(18)
        result = seasonal_naive_forecast(series, holdout_months=6)
        assert len(result) == 6

    def test_naive_values_match_same_month_last_year(self):
        # 18개월: y=1..18, holdout=마지막 6개월(y=13..18)
        # 전년동월 = y=1..6
        series = self._make_series(18)
        result = seasonal_naive_forecast(series, holdout_months=6)
        assert list(result["yhat"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_output_columns(self):
        series = self._make_series(18)
        result = seasonal_naive_forecast(series, holdout_months=6)
        assert set(result.columns) >= {"ds", "y", "yhat"}

    def test_invalid_holdout_raises(self):
        series = self._make_series(6)
        with pytest.raises(ValueError):
            seasonal_naive_forecast(series, holdout_months=6)


class TestBuildComparisonTable:
    def test_sorted_by_mape(self):
        results = {
            "모델A": {"RMSE": 10.0, "MAE": 8.0, "MAPE(%)": 15.0},
            "모델B": {"RMSE": 5.0, "MAE": 4.0, "MAPE(%)": 8.0},
        }
        df = build_comparison_table(results)
        assert df.iloc[0]["모델"] == "모델B"

    def test_row_count(self):
        results = {
            "A": {"RMSE": 1.0, "MAE": 1.0, "MAPE(%)": 1.0},
            "B": {"RMSE": 2.0, "MAE": 2.0, "MAPE(%)": 2.0},
        }
        df = build_comparison_table(results)
        assert len(df) == 2
