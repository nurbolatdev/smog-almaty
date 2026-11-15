"""Сравнение модели CAMS с измерениями станции (гипотеза H4, этап 5)."""

import pandas as pd

from src import features

PAIR_COLUMNS = ["time_utc", "pm25", "cams_pm25", "heating_season"]


def paired_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """Измеренные часы станции, где есть и PM2.5 станции, и прогноз CAMS."""
    required = set(PAIR_COLUMNS + ["pm25_interpolated"])
    assert required.issubset(hourly.columns), "в таблице часов не хватает нужных колонок"
    assert hourly["time_utc"].is_unique, "метки time_utc должны быть уникальны"
    assert hourly["pm25_interpolated"].notna().all(), "в pm25_interpolated есть пропуски"
    # станция и CAMS уже соединены в одной строке по time_utc на этапе 3.1, поэтому
    # повторный merge не нужен; берём только реальные (не интерполированные) измерения
    measured = (
        hourly["pm25"].notna()
        & hourly["cams_pm25"].notna()
        & ~hourly["pm25_interpolated"].astype(bool)
    )
    pairs = hourly.loc[measured, PAIR_COLUMNS].copy()
    pairs["diff"] = pairs["cams_pm25"] - pairs["pm25"]
    return pairs.reset_index(drop=True)


def daily_pairs(
    pairs: pd.DataFrame,
    calendar: pd.DataFrame,
    min_hours: int = features.MIN_HOURS_PER_DAY,
) -> pd.DataFrame:
    """Суточные средние по местной дате из общих часов, выровненные по календарю."""
    assert {"time_utc", "pm25", "cams_pm25"}.issubset(pairs.columns), (
        "в pairs не хватает нужных колонок"
    )
    assert pairs[["time_utc", "pm25", "cams_pm25"]].notna().all().all(), "в pairs есть пропуски"
    assert {"date", "heating_season"}.issubset(calendar.columns), (
        "в calendar не хватает date или heating_season"
    )
    assert min_hours >= 1, "min_hours должен быть >= 1"
    # сутки считаем по местной дате Алматы, хотя ключ остаётся UTC;
    # обе стороны усредняем по одним и тем же часам при пороге min_hours (18)
    grouped = (
        pairs.groupby(features.local_dates(pairs["time_utc"]))
        .agg(
            pair_hours=("time_utc", "size"),
            pm25=("pm25", "mean"),
            cams_pm25=("cams_pm25", "mean"),
        )
        .reset_index()
    )
    enough = grouped["pair_hours"] >= min_hours
    grouped["pm25"] = grouped["pm25"].where(enough)
    grouped["cams_pm25"] = grouped["cams_pm25"].where(enough)
    assert grouped.loc[enough, "date"].isin(calendar["date"]).all(), (
        "сутки со средним отсутствуют в календаре"
    )
    # возвращаем все даты календаря подряд — это нужно для лагов и прореживания;
    # validate="one_to_one" и проверка длины страхуют от дублей дат при объединении
    out = calendar[["date", "heating_season"]].merge(
        grouped, on="date", how="left", validate="one_to_one"
    )
    assert len(out) == len(calendar), "merge с календарём изменил число суток"
    out["diff"] = out["cams_pm25"] - out["pm25"]
    return out
