import numpy as np
import pandas as pd
import pytest

from src import cams


def test_paired_hours_keeps_measured_pairs_only():
    hourly = pd.DataFrame(
        {
            "time_utc": pd.date_range("2023-01-01 00:00", periods=5, freq="h", tz="UTC"),
            "pm25": [10.0, np.nan, 12.0, 15.0, 20.0],
            "cams_pm25": [6.0, 8.0, np.nan, 11.0, 23.0],
            "pm25_interpolated": [False, False, False, True, False],
            "heating_season": [True, True, True, True, False],
        }
    )

    result = cams.paired_hours(hourly)

    assert list(result.columns) == ["time_utc", "pm25", "cams_pm25", "heating_season", "diff"]
    assert len(result) == 2
    assert result["diff"].tolist() == [-4.0, 3.0]


def test_paired_hours_rejects_duplicate_times():
    dup_time = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    hourly = pd.DataFrame(
        {
            "time_utc": [dup_time, dup_time],
            "pm25": [10.0, 12.0],
            "cams_pm25": [8.0, 9.0],
            "pm25_interpolated": [False, False],
            "heating_season": [True, True],
        }
    )

    with pytest.raises(AssertionError):
        cams.paired_hours(hourly)


def test_daily_pairs_groups_by_almaty_date():
    # 18:00 UTC 1 января = 00:00 2 января в Алматы (UTC+6); 24 ч + 5 ч следующих суток
    times = pd.date_range("2023-01-01 18:00", periods=29, freq="h", tz="UTC")
    pairs = pd.DataFrame(
        {
            "time_utc": times,
            "pm25": [10.0] * 29,
            "cams_pm25": [4.0] * 29,
        }
    )
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="D"),
            "heating_season": [True, True, True],
        }
    )

    result = cams.daily_pairs(pairs, calendar)

    assert len(result) == 3
    day1 = result[result["date"] == pd.Timestamp("2023-01-01")].iloc[0]
    day2 = result[result["date"] == pd.Timestamp("2023-01-02")].iloc[0]
    day3 = result[result["date"] == pd.Timestamp("2023-01-03")].iloc[0]
    assert pd.isna(day1["pm25"]) and pd.isna(day1["diff"])
    assert day2["pair_hours"] == 24
    assert day2["pm25"] == pytest.approx(10.0)
    assert day2["diff"] == pytest.approx(-6.0)
    assert day3["pair_hours"] == 5
    assert pd.isna(day3["pm25"]) and pd.isna(day3["diff"])


def test_daily_pairs_rejects_date_outside_calendar():
    # 24 часа принадлежат местным суткам 2023-01-03, которых нет в календаре
    pairs = pd.DataFrame(
        {
            "time_utc": pd.date_range("2023-01-02 18:00", periods=24, freq="h", tz="UTC"),
            "pm25": [10.0] * 24,
            "cams_pm25": [5.0] * 24,
        }
    )
    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=2, freq="D"),
            "heating_season": [True, True],
        }
    )

    with pytest.raises(AssertionError):
        cams.daily_pairs(pairs, calendar)


def test_agreement_metrics_constant_shift():
    station = np.arange(1.0, 11.0)
    df = pd.DataFrame({"pm25": station, "cams_pm25": station + 5.0})

    result = cams.agreement_metrics(df)

    assert result["n"] == 10
    assert result["station_mean"] == pytest.approx(5.5)
    assert result["cams_mean"] == pytest.approx(10.5)
    assert result["bias"] == pytest.approx(5.0)
    assert result["bias_pct"] == pytest.approx(5.0 / 5.5 * 100.0)
    assert result["mae"] == pytest.approx(5.0)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)


def test_agreement_metrics_bias_zero_mae_positive_and_drops_nan():
    station = np.array([10.0, 12.0, 14.0, 16.0, 20.0])
    # первые 4 строки: ±2 поровну (bias = 0, MAE = 2); последняя с NaN отбрасывается
    model = np.array([12.0, 10.0, 16.0, 14.0, np.nan])
    df = pd.DataFrame({"pm25": station, "cams_pm25": model})

    result = cams.agreement_metrics(df)

    assert result["n"] == 4
    assert result["bias"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(2.0)


def test_bootstrap_means_constant_and_reproducible():
    constant = pd.Series([7.0] * 10)
    const_means = cams.bootstrap_means(constant, n_boot=50, seed=42)
    assert len(const_means) == 50
    assert (const_means == 7.0).all()

    values = pd.Series(np.arange(1.0, 21.0))
    same_1 = cams.bootstrap_means(values, n_boot=100, seed=42)
    same_2 = cams.bootstrap_means(values, n_boot=100, seed=42)
    other = cams.bootstrap_means(values, n_boot=100, seed=7)
    assert np.array_equal(same_1, same_2)
    assert not np.array_equal(same_1, other)

    with pytest.raises(AssertionError):
        cams.bootstrap_means(pd.Series([1.0, np.nan, 3.0]), n_boot=10, seed=42)
