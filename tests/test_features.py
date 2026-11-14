import pandas as pd
import pytest

from src import clean, features


def test_season_of_month_uses_meteorological_seasons():
    months = pd.Series([12, 1, 2, 3, 5, 6, 8, 9, 11])

    seasons = features.season_of_month(months)

    expected = ["winter"] * 3 + ["spring"] * 2 + ["summer"] * 2 + ["autumn"] * 2
    assert seasons.tolist() == expected


def test_heating_season_flag_boundaries():
    dates = pd.Series(
        pd.to_datetime(
            ["2023-10-14", "2023-10-15", "2024-01-10", "2024-04-15", "2024-04-16", "2024-07-01"]
        )
    )

    flags = features.heating_season_flag(dates)

    assert flags.tolist() == [False, True, True, True, False, False]


def test_heating_season_flag_uses_local_date():
    # 18:00 UTC = 00:00 следующего дня по Алматы (UTC+6)
    times_utc = pd.Series(pd.to_datetime(["2023-10-14 18:00", "2023-10-14 17:00"], utc=True))

    flags = features.heating_season_flag(clean.to_almaty_time(times_utc))

    assert flags.tolist() == [True, False]


def test_calm_flag_is_strict():
    wind_speed = pd.Series([0.0, 0.99, 1.0, 1.5])

    flags = features.calm_flag(wind_speed)

    assert flags.tolist() == [True, True, False, False]


def test_add_calendar_features_weekend_and_season():
    dates = pd.Series(pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04"]))
    df = pd.DataFrame({"value": [1, 2, 3, 4]})

    out = features.add_calendar_features(df, dates)

    assert out["dayofweek"].tolist() == [4, 5, 6, 0]
    assert out["is_weekend"].tolist() == [False, True, True, False]
    assert (out["month"] == 3).all()
    assert (out["season"] == "spring").all()
    assert out["heating_season"].all()
    # входная таблица не изменилась
    assert df.columns.tolist() == ["value"]


def make_sources():
    """Станция 6 часов, погода на час шире с обеих сторон, CAMS со второй половины."""
    station_hours = pd.date_range("2024-02-29 15:00", periods=6, freq="h", tz="UTC")
    weather_hours = pd.date_range("2024-02-29 14:00", periods=8, freq="h", tz="UTC")
    cams_hours = pd.date_range("2024-02-29 17:00", periods=5, freq="h", tz="UTC")
    pm25 = pd.DataFrame(
        {
            "time_utc": station_hours,
            "pm25": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "pm25_missing": False,
            "pm25_interpolated": False,
        }
    )
    weather = pd.DataFrame(
        {
            "time_utc": weather_hours,
            "temp_c": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
            "humidity": [80, 81, 82, 83, 84, 85, 86, 87],
            "precip": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "pressure": [1010.0, 1011.0, 1012.0, 1013.0, 1014.0, 1015.0, 1016.0, 1017.0],
            "wind_speed": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "wind_dir": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )
    cams = pd.DataFrame({"time_utc": cams_hours, "cams_pm25": [5.0, 6.0, 7.0, 8.0, 9.0]})
    return pm25, weather, cams


def test_align_precip_takes_next_hour_sum():
    hours = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    weather = pd.DataFrame({"time_utc": hours, "precip": [0.0, 1.5, 0.2], "temp_c": [1, 2, 3]})

    aligned = features.align_precip_to_hour_start(weather)

    assert aligned["precip"].iloc[:2].tolist() == [1.5, 0.2]
    assert aligned["precip"].isna().tolist() == [False, False, True]
    assert aligned["temp_c"].tolist() == [1, 2, 3]
    # входная таблица не изменилась
    assert weather["precip"].tolist() == [0.0, 1.5, 0.2]


def test_align_precip_rejects_missing_hour():
    hours = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"], utc=True)
    weather = pd.DataFrame({"time_utc": hours, "precip": [0.0, 1.5, 0.2]})

    with pytest.raises(AssertionError, match="пропавшие часы"):
        features.align_precip_to_hour_start(weather)


def test_merge_sources_keeps_station_hours():
    pm25, weather, cams = make_sources()

    merged = features.merge_sources(pm25, weather, cams)

    assert len(merged) == 6
    # погода берётся по часу UTC, а не по позиции: станция начинается на час позже погоды
    assert merged["temp_c"].tolist() == [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]
    assert merged["cams_pm25"].isna().tolist() == [True, True, False, False, False, False]
    assert merged["cams_pm25"].iloc[2:].tolist() == [5.0, 6.0, 7.0, 8.0]


def test_merge_sources_rejects_duplicate_weather_hour():
    pm25, weather, cams = make_sources()
    weather_with_repeat = pd.concat([weather, weather.iloc[[2]]], ignore_index=True)

    with pytest.raises(pd.errors.MergeError):
        features.merge_sources(pm25, weather_with_repeat, cams)


def test_make_hourly_uses_local_time():
    pm25, weather, cams = make_sources()
    # 15:00 UTC = 21:00, 17:00 = 23:00, 18:00 = 23:00 (смена пояса), 19:00 UTC = 00:00 по Алматы
    merged = features.merge_sources(pm25, weather, cams)

    hourly = features.make_hourly(merged)

    assert hourly["hour"].tolist() == [21, 22, 23, 23, 0, 1]
    assert hourly["month"].tolist() == [2, 2, 2, 2, 3, 3]
    assert hourly.columns.tolist() == features.HOURLY_COLUMNS


def test_check_hourly_accepts_good_table():
    pm25, weather, cams = make_sources()

    hourly = features.make_hourly(features.merge_sources(pm25, weather, cams))

    features.check_hourly(hourly)


def test_check_hourly_rejects_cams_gap_after_start():
    pm25, weather, cams = make_sources()
    hourly = features.make_hourly(features.merge_sources(pm25, weather, cams))
    hourly.loc[4, "cams_pm25"] = float("nan")

    with pytest.raises(AssertionError, match="CAMS"):
        features.check_hourly(hourly)


def test_check_hourly_rejects_missing_weather():
    pm25, weather, cams = make_sources()
    hourly = features.make_hourly(features.merge_sources(pm25, weather, cams))
    hourly.loc[1, "temp_c"] = float("nan")

    with pytest.raises(AssertionError):
        features.check_hourly(hourly)
