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


def test_heating_season_label_boundaries():
    local_times = pd.Series(
        pd.to_datetime(["2020-10-15", "2021-04-15", "2021-12-31", "2022-01-01"]).tz_localize(
            "Asia/Almaty"
        )
    )

    labels = eda.heating_season_label(local_times)

    assert labels.tolist() == ["2020/21", "2020/21", "2021/22", "2021/22"]


def test_hour_profile_table_mean_and_count():
    df = pd.DataFrame(
        {
            "hour": [0, 0, 0, 1, 0, 1],
            "group": ["a", "a", "a", "a", "b", "b"],
            "value": [10.0, 20.0, np.nan, 5.0, 30.0, 40.0],
        }
    )

    mean_table = eda.hour_profile_table(df, "value", "group")
    count_table = eda.hour_profile_table(df, "value", "group", aggfunc="count")

    assert mean_table.loc[0, "a"] == 15.0
    assert count_table.loc[0, "a"] == 2
    assert mean_table.columns.tolist() == ["a", "b"]


def test_smooth_profile_wraps_around_midnight():
    profile = pd.Series(0.0, index=range(24))
    profile.loc[0] = 3.0

    smoothed = eda.smooth_profile(profile)

    assert smoothed.loc[23] == pytest.approx(1.0)
    assert smoothed.loc[0] == pytest.approx(1.0)
    assert smoothed.loc[1] == pytest.approx(1.0)
    assert (smoothed.loc[2:22] == 0.0).all()

    with pytest.raises(AssertionError):
        eda.smooth_profile(profile.iloc[:23])


def test_smooth_profile_rejects_nan():
    profile = pd.Series(0.0, index=range(24))
    profile.loc[5] = np.nan

    with pytest.raises(AssertionError):
        eda.smooth_profile(profile)


def test_profile_extremes_ignores_single_hour_spike():
    hours = np.arange(24)
    values = np.cos(2 * np.pi * (hours - 14) / 24)
    # одиночный скачок в 3 ч выше настоящего пика в 14 ч
    values[3] = 1.5
    table = pd.DataFrame({"season": pd.Series(values, index=hours)})

    assert table["season"].idxmax() == 3

    extremes = eda.profile_extremes(table)

    # ради этого сглаживаем: одиночный скачок не перебивает широкий пик в 14 ч
    assert extremes.loc["season", "peak_hour"] == 14


def test_profile_shift_finds_one_hour_earlier_peak():
    hours = np.arange(24)
    reference = pd.Series(np.cos(2 * np.pi * (hours - 14) / 24), index=hours)
    other = pd.Series(np.roll(reference.to_numpy(), -1), index=hours)

    table = eda.profile_shift(reference, other)

    assert len(table) == 13
    best = table.loc[table["corr"].idxmax()]
    assert best["shift"] == -1
    assert best["corr"] == pytest.approx(1.0)


def test_profile_shift_rejects_nan():
    hours = np.arange(24)
    valid = pd.Series(np.cos(2 * np.pi * hours / 24), index=hours)
    with_nan = valid.copy()
    with_nan.loc[0] = np.nan

    with pytest.raises(AssertionError):
        eda.profile_shift(with_nan, valid)
    with pytest.raises(AssertionError):
        eda.profile_shift(valid, with_nan)


def _toy_lagged_df() -> pd.DataFrame:
    time_utc = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    x = pd.Series(np.random.default_rng(0).normal(size=48))
    return pd.DataFrame({"time_utc": time_utc, "x": x, "y": x.shift(2)})


def test_lagged_correlations_finds_delay():
    df = _toy_lagged_df()

    table = eda.lagged_correlations(df, "y", "x", range(-3, 4))

    best = table.loc[table["rho"].idxmax()]
    assert best["lag"] == 2
    assert best["rho"] == pytest.approx(1.0)
    assert best["n"] == 46


def test_lagged_correlations_rows_mask_and_grid_check():
    df = _toy_lagged_df()
    rows = pd.Series(df.index < 24, index=df.index)

    table = eda.lagged_correlations(df, "y", "x", [0], rows=rows)

    # у y первые два часа NaN, поэтому из первых 24 строк остаются 22 пары
    assert table.loc[0, "n"] == 22

    with pytest.raises(AssertionError):
        eda.lagged_correlations(df.drop(index=10), "y", "x", [0])
