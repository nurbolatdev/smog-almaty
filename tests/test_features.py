import pandas as pd

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
