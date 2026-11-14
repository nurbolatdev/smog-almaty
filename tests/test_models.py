import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

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


def test_linear_model_scaler_fitted_on_train_only():
    X_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [10.0, 20.0, 30.0, 40.0]})
    y_train = pd.Series([2.0, 4.0, 6.0, 8.0])
    X_test = X_train + 1000.0

    model = models.make_linear_model()
    model.fit(X_train, y_train)
    _ = model.predict(X_test)

    scale_mean = model.named_steps["scale"].mean_
    assert np.allclose(scale_mean, X_train.mean().to_numpy())


def test_linear_coefficients_recover_true_slopes():
    rng = np.random.default_rng(42)
    x1 = rng.normal(0.0, 1.0, size=50)
    x2 = rng.normal(0.0, 5.0, size=50)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = 3.0 * x1 - 2.0 * x2 + 5.0

    model = models.make_linear_model()
    model.fit(X, y)
    coef = models.linear_coefficients(model, ["x1", "x2"]).set_index("feature")

    # по модулю |coef_std| у x2 (~10) больше, чем у x1 (~3), поэтому x2 должна идти первой
    assert coef.index.tolist() == ["x2", "x1"]
    assert coef.loc["x1", "coef_original"] == pytest.approx(3.0)
    assert coef.loc["x2", "coef_original"] == pytest.approx(-2.0)
    assert coef.loc["x1", "coef_std"] == pytest.approx(3.0 * x1.std(ddof=0))
    assert coef.loc["x2", "coef_std"] == pytest.approx(-2.0 * x2.std(ddof=0))


def test_make_forest_reproducible():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=40), "x2": rng.normal(size=40)})
    y = 2.0 * X["x1"] + rng.normal(scale=0.1, size=40)

    forest_a = models.make_forest(n_estimators=20, max_depth=4, min_samples_leaf=3, seed=42)
    forest_b = models.make_forest(n_estimators=20, max_depth=4, min_samples_leaf=3, seed=42)
    forest_a.fit(X, y)
    forest_b.fit(X, y)

    assert forest_a.n_estimators == 20
    assert forest_a.max_depth == 4
    assert forest_a.min_samples_leaf == 3
    assert np.allclose(forest_a.predict(X), forest_b.predict(X))


def test_permutation_table_informative_feature_first():
    rng = np.random.default_rng(7)
    x1 = np.linspace(0.0, 10.0, 60)
    x2 = rng.normal(size=60)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = 4.0 * x1

    model = models.make_linear_model()
    model.fit(X, y)
    imp = models.permutation_table(model, X, y, n_repeats=20, seed=42)

    assert imp.columns.tolist() == ["feature", "mae_increase", "mae_increase_std"]
    assert imp["feature"].tolist() == ["x1", "x2"]
    assert imp.loc[0, "mae_increase"] > 0.0
    assert abs(imp.loc[1, "mae_increase"]) < 0.1 * imp.loc[0, "mae_increase"]


def test_time_cv_table_folds_are_chronological():
    x = np.arange(60, dtype=float)
    y = 2.0 * x + 1.0
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "x": x,
            "pm25_lag1": pd.Series(y).shift(1),
            "pm25_mean": y,
        }
    )
    table = models.make_model_table(raw, ["x", "pm25_lag1"])
    model = LinearRegression()

    cv = models.time_cv_table(table, ["x"], model, n_splits=5)

    assert len(cv) == 5
    assert (cv["train_end"] < cv["val_start"]).all()
    assert (cv["mae_model"] < 1e-9).all()
    assert (cv["mae_naive"] > 0.0).all()
    assert cv["n_val"].sum() == len(table) - cv.loc[0, "n_train"]
    # исходная модель должна остаться необученной благодаря clone
    assert not hasattr(model, "coef_")


def test_make_forest_grid_size():
    grid = models.make_forest_grid([100, 300], [4, 8], [1, 5])

    assert len(grid) == 8
    keys = {"n_estimators", "max_depth", "min_samples_leaf"}
    assert all(set(item.keys()) == keys for item in grid)
    unique_items = {tuple(sorted(item.items())) for item in grid}
    assert len(unique_items) == 8


def test_forest_grid_cv_sorted():
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 10.0, 50)
    y = 3.0 * x + rng.normal(scale=0.5, size=50)
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=50, freq="D"),
            "x": x,
            "pm25_lag1": pd.Series(y).shift(1),
            "pm25_mean": y,
        }
    )
    table = models.make_model_table(raw, ["x", "pm25_lag1"])
    grid = models.make_forest_grid([10, 20], [2, 6], [1, 5])

    res = models.forest_grid_cv(table, ["x", "pm25_lag1"], grid, n_splits=3, seed=42)

    assert len(res) == len(grid)
    assert res["cv_mae_mean"].is_monotonic_increasing
    for col in ["n_estimators", "max_depth", "min_samples_leaf"]:
        assert pd.api.types.is_integer_dtype(res[col])


def test_residuals_on_calendar_fills_gaps():
    calendar = pd.date_range("2024-03-01", periods=5, freq="D")
    dates = pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-04", "2024-03-05"])
    residuals = np.array([1.0, -2.0, 3.0, -1.0])

    full = models.residuals_on_calendar(calendar, dates, residuals)

    assert len(full) == 5
    assert full.columns.tolist() == ["date", "residual"]
    assert pd.isna(full.loc[2, "residual"])
    assert full.loc[[0, 1, 3, 4], "residual"].tolist() == [1.0, -2.0, 3.0, -1.0]

    dup_dates = pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-04", "2024-03-05"])
    with pytest.raises(pd.errors.MergeError):
        models.residuals_on_calendar(calendar, dup_dates, residuals)
