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


def check_clean_hourly(df: pd.DataFrame, columns: list[str]) -> None:
    """Контракт для объединения (3.1): один час UTC — одна строка, без пропусков."""
    check_columns(df, columns)
    times = df["time_utc"]
    assert isinstance(times.dtype, pd.DatetimeTZDtype), "time_utc не дата с часовым поясом"
    assert str(times.dt.tz) == "UTC", "time_utc не в UTC"
    assert times.isna().sum() == 0, "в time_utc есть пропуски"
    assert times.is_unique, "есть повторяющиеся метки"
    assert times.is_monotonic_increasing, "метки не по порядку"
    one_hour = times.diff().dropna() == pd.Timedelta(hours=1)
    assert one_hour.all(), "шаг между метками не 1 час: есть пропавшие часы"
    for column in columns[1:]:
        assert pd.api.types.is_numeric_dtype(df[column]), f"{column} не число"
        assert df[column].isna().sum() == 0, f"в {column} есть пропуски"
    out_of_range = count_out_of_range(df, VALID_RANGES)
    assert sum(out_of_range.values()) == 0, f"значения вне диапазонов: {out_of_range}"


def save_clean_csv(df: pd.DataFrame, path: str) -> None:
    """Записывает очищенную таблицу; файл перезаписывается при каждом запуске."""
    df.to_csv(path, index=False, date_format=PROCESSED_TIME_FORMAT)


def read_clean_csv(path: str) -> pd.DataFrame:
    """Читает очищенную таблицу так, как это делают 2.2 и 3.1."""
    df = pd.read_csv(path)
    df["time_utc"] = parse_utc_times(df["time_utc"], PROCESSED_TIME_FORMAT)
    return df
