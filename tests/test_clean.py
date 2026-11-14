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
