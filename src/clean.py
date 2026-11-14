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


# Физически возможные значения, границы включены: всё вне — ошибка данных, а не выброс.
VALID_RANGES = {
    "temp_c": (-50, 50),  # °C
    "humidity": (0, 100),  # %
    "precip": (0, 100),  # мм/ч
    "pressure": (900, 1100),  # гПа
    "wind_speed": (0, 60),  # м/с
    "wind_dir": (0, 360),  # градусы
    "cams_pm25": (0, 1000),  # µg/m³
}


def check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Проверяет, что колонки таблицы — ровно ожидаемые и в том же порядке."""
    assert list(df.columns) == columns, f"колонки {list(df.columns)}, ожидали {columns}"


def count_out_of_range(df: pd.DataFrame, valid_ranges: dict) -> dict:
    """Считает значения вне допустимых границ по каждой колонке таблицы."""
    counts = {}
    for column, (low, high) in valid_ranges.items():
        if column not in df.columns:
            continue  # одна константа на погоду и CAMS: чужих колонок в таблице нет
        # NaN не нарушение: сравнение с NaN даёт False
        outside = (df[column] < low) | (df[column] > high)
        counts[column] = int(outside.sum())
    return counts


def iqr_outlier_mask(values: pd.Series, k: float = 1.5) -> pd.Series:
    """Помечает выбросы по IQR: только помечает, ничего не удаляет."""
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    return (values < q1 - k * iqr) | (values > q3 + k * iqr)
