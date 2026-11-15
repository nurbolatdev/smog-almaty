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
