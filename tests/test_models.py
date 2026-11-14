import numpy as np
import pandas as pd
import pytest

from src import models


def _sample_daily() -> pd.DataFrame:
    """Игрушечные 10 суток подряд с одним NaN в цели и одним в лаге."""
    pm25 = pd.Series([10.0, 12.0, np.nan, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0])
    lag1 = pd.Series([np.nan, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0])
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "temp_c": np.linspace(-5.0, 4.0, 10),
            "heating_season": [True] * 5 + [False] * 5,
            "pm25_lag1": lag1,
            "pm25_mean": pm25,
        }
    )


def test_make_model_table_drops_missing_and_casts():
    features = ["temp_c", "heating_season", "pm25_lag1"]
    table = models.make_model_table(_sample_daily(), features)

    assert len(table) == 8
    assert not table.isna().any().any()
    assert table.columns.tolist() == ["date", *features, "pm25_mean"]
    assert table["heating_season"].dtype == float
    assert set(table["heating_season"].unique()) == {0.0, 1.0}


def test_split_by_date_boundaries():
    full = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "temp_c": np.arange(10, dtype=float),
            "pm25_mean": np.arange(10.0, 20.0),
        }
    )
    train, test = models.split_by_date(full, "2024-01-07")

    assert len(train) == 6
    assert len(test) == 4
    assert train["date"].max() < test["date"].min()

    with pytest.raises(AssertionError):
        models.split_by_date(full, "2024-01-01")


def test_naive_and_month_mean_forecast():
    train = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-15", "2024-02-05", "2024-02-15"]),
            "pm25_lag1": [9.0, 11.0, 19.0, 21.0],
            "pm25_mean": [10.0, 20.0, 30.0, 50.0],
        }
    )
    test = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-10", "2025-02-10"]),
            "pm25_lag1": [15.0, 35.0],
            "pm25_mean": [12.0, 42.0],
        },
        index=[10, 20],
    )

    naive = models.naive_forecast(test)
    assert naive.tolist() == [15.0, 35.0]

    monthly = models.month_mean_forecast(train, test)
    assert monthly.index.equals(test.index)
    assert monthly.tolist() == [15.0, 40.0]

    missing_month_test = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-01"]),
            "pm25_lag1": [10.0],
            "pm25_mean": [12.0],
        }
    )
    with pytest.raises(AssertionError):
        models.month_mean_forecast(train, missing_month_test)


def test_regression_metrics_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 8.0])

    res = models.regression_metrics(y, pred)
    assert res["n"] == 4
    assert res["mae"] == pytest.approx(1.0)
    assert res["rmse"] == pytest.approx(2.0)
    assert res["r2"] == pytest.approx(-2.2)

    perfect = models.regression_metrics(y, y)
    assert perfect["r2"] == pytest.approx(1.0)


def test_bootstrap_mae_diff_paired():
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    pred_b = np.array([12.0, 18.0, 35.0, 36.0, 55.0])

    perfect_diffs = models.bootstrap_mae_diff(y, y, pred_b, n_boot=200, seed=42)
    assert (perfect_diffs <= 0.0).all()

    equal_diffs = models.bootstrap_mae_diff(y, pred_b, pred_b, n_boot=200, seed=42)
    assert (equal_diffs == 0.0).all()

    same_1 = models.bootstrap_mae_diff(y, y + 1.0, pred_b, n_boot=100, seed=7)
    same_2 = models.bootstrap_mae_diff(y, y + 1.0, pred_b, n_boot=100, seed=7)
    assert np.array_equal(same_1, same_2)
