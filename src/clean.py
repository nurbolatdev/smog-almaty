"""Очистка почасовых таблиц: время, проверки значений, выбросы, запись очищенных файлов."""

import pandas as pd

# Местное время Алматы: UTC+6 до 2024-03-01, затем UTC+5.
# Смену пояса берёт база часовых поясов, вручную смещения не считаем.
LOCAL_TZ = "Asia/Almaty"

# Формат времени в очищенных CSV: явное смещение +0000 — в файле видно, что это UTC.
PROCESSED_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def parse_utc_times(times: pd.Series, time_format: str) -> pd.Series:
    """Разбирает строки в время UTC и проверяет, что битых меток нет."""
    # coerce превращает битые метки в NaT, чтобы посчитать все сразу, а не упасть на первой
    parsed = pd.to_datetime(times, utc=True, format=time_format, errors="coerce")
    nat_count = parsed.isna().sum()
    assert nat_count == 0, f"не удалось разобрать {nat_count} меток времени"
    return parsed


def to_almaty_time(times_utc: pd.Series) -> pd.Series:
    """Переводит время UTC в Asia/Almaty для признаков и графиков (ключ merge — time_utc)."""
    assert str(times_utc.dt.tz) == "UTC", "ожидаем метки в UTC"
    return times_utc.dt.tz_convert(LOCAL_TZ)
