"""Объединение источников и признаки: часовая и суточная таблицы (этап 3)."""

import pandas as pd

from src import clean

# Суточная норма ВОЗ (2021) для PM2.5, µg/m³
WHO_DAILY_LIMIT = 15

# Штиль: скорость ветра строго меньше этого порога, м/с
CALM_WIND_SPEED = 1.0

# Минимум часов со значением PM2.5 для суточного среднего (18 и для 25-часовых суток 29.02.2024)
MIN_HOURS_PER_DAY = 18

# Метеорологические сезоны по номеру месяца
SEASON_BY_MONTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

HOURLY_COLUMNS = [
    "time_utc",
    "pm25",
    "pm25_missing",
    "pm25_interpolated",
    "temp_c",
    "humidity",
    "precip",
    "pressure",
    "wind_speed",
    "wind_dir",
    "cams_pm25",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "season",
    "heating_season",
    "is_calm",
]

DAILY_COLUMNS = [
    "date",
    "pm25_mean",
    "pm25_hours",
    "pm25_lag1",
    "who_exceed",
    "temp_c",
    "humidity",
    "precip",
    "pressure",
    "wind_speed",
    "dayofweek",
    "month",
    "is_weekend",
    "season",
    "heating_season",
    "is_calm",
]

# Формат даты в daily.csv
DAILY_DATE_FORMAT = "%Y-%m-%d"


def season_of_month(months: pd.Series) -> pd.Series:
    """Метеорологический сезон по номеру месяца."""
    return months.map(SEASON_BY_MONTH)


def heating_season_flag(times: pd.Series) -> pd.Series:
    """Отопительный сезон: с 15 октября по 15 апреля включительно (по местной дате)."""
    month = times.dt.month
    day = times.dt.day
    whole_months = (month >= 11) | (month <= 3)
    october_part = (month == 10) & (day >= 15)
    april_part = (month == 4) & (day <= 15)
    return whole_months | october_part | april_part


def calm_flag(wind_speed: pd.Series) -> pd.Series:
    """Штиль: ветер строго меньше 1 м/с."""
    return wind_speed < CALM_WIND_SPEED


def add_calendar_features(df: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    """Добавляет dayofweek, month, is_weekend, season, heating_season по местному времени times."""
    out = df.copy()
    # 0 = понедельник, как в pandas
    out["dayofweek"] = times.dt.day_of_week
    out["month"] = times.dt.month
    out["is_weekend"] = out["dayofweek"] >= 5
    out["season"] = season_of_month(out["month"])
    out["heating_season"] = heating_season_flag(times)
    return out


def align_precip_to_hour_start(weather: pd.DataFrame) -> pd.DataFrame:
    """Осадки в строке T — за час [T, T+1 ч), как PM2.5 станции."""
    # shift на строку = shift на час только на полной сетке
    steps = weather["time_utc"].diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all(), "в погоде пропавшие часы: shift(-1) ошибётся"
    aligned = weather.copy()
    # в Open-Meteo осадки в метке T — сумма за предыдущий час, берём значение из метки T+1 ч
    aligned["precip"] = weather["precip"].shift(-1)
    return aligned


def merge_sources(pm25: pd.DataFrame, weather: pd.DataFrame, cams: pd.DataFrame) -> pd.DataFrame:
    """Объединяет по time_utc: строки — часы станции; CAMS до своего начала пустой."""
    merged = pm25.merge(weather, on="time_utc", how="left", validate="one_to_one")
    assert len(merged) == len(pm25), "merge с погодой изменил число строк"
    merged = merged.merge(cams, on="time_utc", how="left", validate="one_to_one")
    assert len(merged) == len(pm25), "merge с CAMS изменил число строк"
    return merged


def make_hourly(merged: pd.DataFrame) -> pd.DataFrame:
    """Часовая таблица: признаки по местному времени, ключ остаётся time_utc."""
    local = clean.to_almaty_time(merged["time_utc"])
    hourly = add_calendar_features(merged, local)
    hourly["hour"] = local.dt.hour
    hourly["is_calm"] = calm_flag(hourly["wind_speed"])
    return hourly[HOURLY_COLUMNS]


def check_hourly(df: pd.DataFrame) -> None:
    """Контракт hourly.csv."""
    clean.check_columns(df, HOURLY_COLUMNS)
    clean.check_clean_pm25(df[clean.PM25_COLUMNS])
    # season — текст, остальное числа и флаги
    number_columns = [c for c in HOURLY_COLUMNS if c != "season"]
    clean.check_clean_hourly(
        df.drop(columns="season"), number_columns, nan_columns=("pm25", "cams_pm25")
    )
    assert df["season"].isin(set(SEASON_BY_MONTH.values())).all(), "неизвестный сезон"
    # CAMS пустой только до своей первой метки
    first_cams = df.loc[df["cams_pm25"].notna(), "time_utc"].min()
    cams_empty = df["cams_pm25"].isna()
    assert (cams_empty == (df["time_utc"] < first_cams)).all(), (
        "CAMS пустой не только до начала источника"
    )
