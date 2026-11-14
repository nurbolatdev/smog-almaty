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
