"""Объединение источников и признаки: часовая и суточная таблицы (этап 3)."""

import pandas as pd

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
