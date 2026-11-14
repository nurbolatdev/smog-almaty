import numpy as np
import pandas as pd
import pytest

from src import eda


def test_who_exceed_share_skips_unknown_days():
    daily = pd.DataFrame(
        {
            "month": [1, 1, 1, 2, 2],
            "who_exceed": pd.array([True, False, pd.NA, True, True], dtype="boolean"),
        }
    )

    table = eda.who_exceed_share_by_month(daily)

    assert table.loc[0, "month"] == 2
    assert table.loc[0, "share"] == 1.0
    january = table[table["month"] == 1].iloc[0]
    assert january["days_known"] == 2
    assert january["days_exceed"] == 1
    # если бы <NA> считался False, доля была бы 1/3
    assert january["share"] == 0.5


def _toy_hourly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": [0, 0, 0, 1, 0],
            "month": [1, 1, 1, 1, 2],
            "pm25": [10, 20, np.nan, 5, 30],
        }
    )


def test_hour_month_table_mean_ignores_missing_hours():
    table = eda.hour_month_table(_toy_hourly())

    assert table.loc[0, 1] == 15.0
    assert table.index.tolist() == [0, 1]
    assert table.columns.tolist() == [1, 2]


def test_hour_month_table_count_counts_only_values():
    table = eda.hour_month_table(_toy_hourly(), aggfunc="count")

    assert table.loc[0, 1] == 2


def test_mean_with_ci_t_interval():
    df = pd.DataFrame(
        {
            "group": ["a", "a", "a", "a", "b", "b"],
            "value": [1.0, 2.0, 3.0, np.nan, 10.0, 10.0],
        }
    )

    table = eda.mean_with_ci(df, "value", ["group"])

    a = table[table["group"] == "a"].iloc[0]
    assert a["mean"] == 2.0
    assert a["n"] == 3
    assert a["ci_low"] == pytest.approx(-0.48414, abs=1e-4)
    assert a["ci_high"] == pytest.approx(4.48414, abs=1e-4)
    b = table[table["group"] == "b"].iloc[0]
    assert b["ci_low"] == 10.0
    assert b["ci_high"] == 10.0


def test_binned_medians_on_line():
    x = list(range(100)) + [1000] * 5
    y = [2 * value for value in range(100)] + [np.nan] * 5
    df = pd.DataFrame({"x": x, "y": y})

    table = eda.binned_medians(df, "x", "y")

    # строки с NaN убраны до разбиения на интервалы
    assert len(table) == 10
    assert (table["n"] == 10).all()
    assert table.loc[0, "x_median"] == 4.5
    assert table.loc[0, "y_median"] == 9.0


def test_spearman_matrix_monotonic_and_drops_missing():
    x = [1, 2, 3, 4, 5, 6]
    df = pd.DataFrame(
        {
            "x": x,
            # куб монотонен, но не линеен: поэтому Спирмен, а не Пирсон
            "cube": [1, 8, 27, 64, 125, np.nan],
            "neg": [-value for value in x],
        }
    )

    matrix = eda.spearman_matrix(df, ["x", "cube", "neg"])

    assert matrix.loc["x", "cube"] == pytest.approx(1.0)
    assert matrix.loc["x", "neg"] == pytest.approx(-1.0)
    assert matrix.columns.tolist() == ["x", "cube", "neg"]
    assert matrix.index.tolist() == ["x", "cube", "neg"]
