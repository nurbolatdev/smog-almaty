"""Тесты src/clean.py: маленькие таблицы в памяти, сырые файлы не читаем."""

import pandas as pd
import pytest

from src import clean

METEO_FORMAT = "%Y-%m-%dT%H:%M"


def test_parse_utc_times_returns_utc():
    times = pd.Series(["2024-02-29T17:00", "2024-02-29T18:00"])

    result = clean.parse_utc_times(times, METEO_FORMAT)

    assert str(result.dt.tz) == "UTC"
    assert result.tolist() == [
        pd.Timestamp("2024-02-29 17:00", tz="UTC"),
        pd.Timestamp("2024-02-29 18:00", tz="UTC"),
    ]


def test_parse_utc_times_rejects_bad_string():
    times = pd.Series(["not a date"])

    with pytest.raises(AssertionError, match="разобрать 1"):
        clean.parse_utc_times(times, METEO_FORMAT)


def test_parse_utc_times_rejects_other_format():
    times = pd.Series(["2024-02-29T17:00:00Z"])

    with pytest.raises(AssertionError):
        clean.parse_utc_times(times, METEO_FORMAT)


def test_to_almaty_time_handles_march_2024_change():
    times = pd.Series(pd.to_datetime(["2024-02-29 17:00", "2024-02-29 18:00"], utc=True))

    result = clean.to_almaty_time(times)

    # одно и то же местное время дважды — поэтому ключ объединения UTC
    assert result.dt.strftime("%Y-%m-%d %H:%M%z").tolist() == [
        "2024-02-29 23:00+0600",
        "2024-02-29 23:00+0500",
    ]


def test_to_almaty_time_offsets_before_and_after():
    times = pd.Series(pd.to_datetime(["2023-01-15 00:00", "2025-01-15 00:00"], utc=True))

    result = clean.to_almaty_time(times)

    assert result.dt.strftime("%Y-%m-%d %H:%M%z").tolist() == [
        "2023-01-15 06:00+0600",
        "2025-01-15 05:00+0500",
    ]


def test_to_almaty_time_rejects_naive_times():
    times = pd.Series(pd.to_datetime(["2024-02-29 17:00"]))

    with pytest.raises(AssertionError):
        clean.to_almaty_time(times)


def make_weather_table(n=3):
    """Таблица погоды с допустимыми значениями: n часов подряд от 2024-02-29 16:00 UTC."""
    return pd.DataFrame(
        {
            "time_utc": pd.date_range("2024-02-29 16:00", periods=n, freq="h", tz="UTC"),
            "temp_c": [5.0] * n,
            "humidity": [50] * n,
            "precip": [0.0] * n,
            "pressure": [1015.0] * n,
            "wind_speed": [2.0] * n,
            "wind_dir": [180] * n,
        }
    )


def test_check_columns_accepts_expected():
    table = make_weather_table()

    clean.check_columns(table, list(table.columns))


def test_check_columns_rejects_other_order():
    table = make_weather_table()
    expected = list(table.columns)
    shuffled = table[["time_utc", "humidity", "temp_c", "precip", "pressure", "wind_speed",
                      "wind_dir"]]

    with pytest.raises(AssertionError):
        clean.check_columns(shuffled, expected)


def test_count_out_of_range_counts_each_column():
    table = make_weather_table()
    table.loc[0, "humidity"] = 101
    table.loc[1, "precip"] = -0.1

    counts = clean.count_out_of_range(table, clean.VALID_RANGES)

    assert counts["humidity"] == 1
    assert counts["precip"] == 1
    assert counts["temp_c"] == 0
    assert "cams_pm25" not in counts


def test_count_out_of_range_keeps_bounds_and_nan():
    table = make_weather_table()
    table["humidity"] = [0.0, 100.0, float("nan")]

    counts = clean.count_out_of_range(table, clean.VALID_RANGES)

    assert counts["humidity"] == 0


def test_iqr_outlier_mask_flags_far_value():
    values = pd.Series([10, 11, 12, 13, 100])

    mask = clean.iqr_outlier_mask(values)

    assert mask.tolist() == [False, False, False, False, True]


def test_iqr_outlier_mask_flags_every_rain_hour_when_iqr_is_zero():
    values = pd.Series([0.0] * 9 + [0.5])

    mask = clean.iqr_outlier_mask(values)

    assert mask.tolist() == [False] * 9 + [True]


def test_check_clean_hourly_accepts_good_table():
    table = make_weather_table()

    clean.check_clean_hourly(table, list(table.columns))


def test_check_clean_hourly_rejects_string_times():
    table = make_weather_table()
    columns = list(table.columns)
    table["time_utc"] = table["time_utc"].astype(str)

    with pytest.raises(AssertionError):
        clean.check_clean_hourly(table, columns)


def test_check_clean_hourly_rejects_missing_hour():
    table = make_weather_table(4)
    columns = list(table.columns)
    table = table.drop(index=1)

    with pytest.raises(AssertionError, match="1 час"):
        clean.check_clean_hourly(table, columns)


def test_check_clean_hourly_rejects_duplicate_hour():
    table = make_weather_table()
    columns = list(table.columns)
    table = pd.concat([table, table.iloc[[2]]], ignore_index=True)

    with pytest.raises(AssertionError):
        clean.check_clean_hourly(table, columns)


def test_check_clean_hourly_rejects_nan_value():
    table = make_weather_table()
    columns = list(table.columns)
    table.loc[1, "temp_c"] = float("nan")

    with pytest.raises(AssertionError):
        clean.check_clean_hourly(table, columns)


def test_check_clean_hourly_rejects_out_of_range():
    table = make_weather_table()
    columns = list(table.columns)
    table.loc[1, "wind_speed"] = -1.0

    with pytest.raises(AssertionError, match="диапазонов"):
        clean.check_clean_hourly(table, columns)


def test_save_and_read_clean_csv_round_trip(tmp_path):
    table = make_weather_table()
    columns = list(table.columns)
    path = str(tmp_path / "weather_clean.csv")

    clean.save_clean_csv(table, path)
    with open(path, encoding="utf-8") as file:
        lines = file.read().splitlines()
    read = clean.read_clean_csv(path)

    assert lines[1].startswith("2024-02-29T16:00:00+0000")
    assert read["time_utc"].tolist() == table["time_utc"].tolist()
    assert str(read["time_utc"].dt.tz) == "UTC"
    pd.testing.assert_frame_equal(
        read.drop(columns="time_utc"), table.drop(columns="time_utc")
    )
    clean.check_clean_hourly(read, columns)


def make_pm25_table():
    """Таблица PM2.5: 8 часов от 2024-02-29 16:00 UTC, дыра из 4 часов в середине."""
    pm25 = pd.Series([10.0, 11.0, None, None, None, None, 12.0, 13.0])
    return pd.DataFrame(
        {
            "time_utc": pd.date_range("2024-02-29 16:00", periods=8, freq="h", tz="UTC"),
            "pm25": pm25,
            "pm25_missing": pm25.isna(),
            "pm25_interpolated": [False, True, False, False, False, False, False, False],
        }
    )


def test_count_out_of_range_counts_pm25():
    table = pd.DataFrame({"pm25": [-1.0, 5.0, 1001.0]})

    counts = clean.count_out_of_range(table, clean.VALID_RANGES)

    assert counts["pm25"] == 2


def test_to_hourly_grid_adds_missing_hour():
    times = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"], utc=True)
    table = pd.DataFrame({"time_utc": times, "pm25": [1.0, 2.0, 4.0]})

    grid = clean.to_hourly_grid(table)

    expected_times = pd.date_range("2024-01-01 00:00", periods=4, freq="h", tz="UTC")
    assert len(grid) == 4
    assert grid["time_utc"].tolist() == expected_times.tolist()
    assert str(grid["time_utc"].dt.tz) == "UTC"
    assert pd.isna(grid.loc[2, "pm25"])
    assert grid["pm25"].dropna().tolist() == [1.0, 2.0, 4.0]


def test_to_hourly_grid_rejects_duplicate_hour():
    times = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00"], utc=True)
    table = pd.DataFrame({"time_utc": times, "pm25": [1.0, 2.0]})

    with pytest.raises(AssertionError, match="повторяющиеся"):
        clean.to_hourly_grid(table)


def test_check_clean_hourly_allows_nan_in_listed_column():
    table = make_weather_table()
    columns = list(table.columns)
    table.loc[1, "temp_c"] = float("nan")

    clean.check_clean_hourly(table, columns, nan_columns=("temp_c",))


def test_check_clean_pm25_accepts_good_table():
    clean.check_clean_pm25(make_pm25_table())


def test_check_clean_pm25_rejects_wrong_missing_flag():
    table = make_pm25_table()
    table.loc[2, "pm25_missing"] = False

    with pytest.raises(AssertionError, match="pm25_missing"):
        clean.check_clean_pm25(table)


def test_check_clean_pm25_rejects_text_flag():
    table = make_pm25_table()
    table["pm25_interpolated"] = ["True", "False"] * 4

    with pytest.raises(AssertionError):
        clean.check_clean_pm25(table)


def test_gap_lengths_marks_each_missing_hour():
    values = pd.Series([1, None, 2, None, None, 3])

    assert clean.gap_lengths(values).tolist() == [0, 1, 0, 2, 2, 0]


def test_gap_sizes_lists_each_gap_once():
    values = pd.Series([1, None, 2, None, None, 3])

    assert clean.gap_sizes(values).tolist() == [1, 2]


def test_interpolate_short_gaps_fills_only_whole_short_gaps():
    nan = float("nan")
    values = pd.Series([1, nan, nan, nan, 5, nan, nan, nan, nan, 10], dtype=float)
    expected = pd.Series([1, 2, 3, 4, 5, nan, nan, nan, nan, 10], dtype=float)

    result = clean.interpolate_short_gaps(values)

    pd.testing.assert_series_equal(result, expected)
    # ловушка: limit=3 заполнил бы 3 из 4 часов длинной дыры
    assert values.interpolate(limit=3).isna().sum() == 1


def test_interpolate_short_gaps_keeps_edges():
    nan = float("nan")
    values = pd.Series([nan, 1, 2, nan], dtype=float)

    result = clean.interpolate_short_gaps(values)

    assert result.isna().tolist() == [True, False, False, True]


def test_interpolate_short_gaps_does_not_change_input():
    nan = float("nan")
    values = pd.Series([1, nan, 3], dtype=float)

    clean.interpolate_short_gaps(values)

    assert values.isna().tolist() == [False, True, False]


def test_stuck_value_mask_flags_long_run():
    long_run = pd.Series([5] * 6 + [7, 7])
    short_run = pd.Series([5] * 5 + [7])

    assert clean.stuck_value_mask(long_run).tolist() == [True] * 6 + [False, False]
    assert not clean.stuck_value_mask(short_run).any()


def test_stuck_value_mask_run_breaks_on_missing_hour():
    values = pd.Series([3, 3, 3, None, 3, 3, 3])

    assert not clean.stuck_value_mask(values).any()


def test_check_clean_pm25_rejects_short_gap_left():
    table = make_pm25_table()
    table.loc[7, "pm25"] = None
    table.loc[7, "pm25_missing"] = True

    with pytest.raises(AssertionError, match="3 ч"):
        clean.check_clean_pm25(table)
