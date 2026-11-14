import numpy as np
import pandas as pd

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
