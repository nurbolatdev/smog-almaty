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
    "pm25": (0, 1000),  # µg/m³, станция
    "cams_pm25": (0, 1000),  # µg/m³
}

# Колонки pm25_clean.csv.
PM25_COLUMNS = ["time_utc", "pm25", "pm25_missing", "pm25_interpolated"]

# Заполняем только дыры целиком не длиннее 3 ч (правило L4).
MAX_INTERPOLATION_HOURS = 3

# Одно и то же значение 6 ч подряд и дольше считаем застывшим датчиком (шаг 19 ноутбука).
STUCK_MIN_HOURS = 6


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


def to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Полная сетка часов от первой до последней метки: пропавший час — строка с NaN."""
    assert df["time_utc"].is_unique, "есть повторяющиеся метки"
    assert df["time_utc"].is_monotonic_increasing, "метки не по порядку"
    # bool-колонки сюда не передаём: у новых строк в них был бы NaN
    grid = df.set_index("time_utc").asfreq("h")
    return grid.reset_index()


def run_ids(values: pd.Series) -> pd.Series:
    """Номер серии: увеличивается, когда значение отличается от предыдущего часа."""
    # NaN != NaN, поэтому каждый пустой час в ряду значений — своя серия
    return (values != values.shift()).cumsum()


def gap_lengths(values: pd.Series) -> pd.Series:
    """Для каждого часа — длина дыры, в которой он стоит (0 у часов со значением)."""
    is_missing = values.isna()
    # у серии часов со значением сумма True равна 0
    return is_missing.groupby(run_ids(is_missing)).transform("sum")


def gap_sizes(values: pd.Series) -> pd.Series:
    """Длины всех дыр: одна строка — одна дыра."""
    is_missing = values.isna()
    sizes = is_missing.groupby(run_ids(is_missing)).sum()
    return sizes[sizes > 0].reset_index(drop=True)


# interpolate(limit=3) заполнил бы первые 3 часа и длинной дыры, а нам нужно
# не трогать длинную дыру целиком.
def interpolate_short_gaps(values: pd.Series, max_hours: int = MAX_INTERPOLATION_HOURS):
    """Линейно заполняет только дыры целиком не длиннее max_hours; ряд — на полной сетке."""
    # края ряда не заполняем
    filled = values.interpolate(limit_area="inside")
    lengths = gap_lengths(values)
    is_short_gap = (lengths > 0) & (lengths <= max_hours)
    return values.where(~is_short_gap, filled)


def stuck_value_mask(values: pd.Series, min_hours: int = STUCK_MIN_HOURS) -> pd.Series:
    """Помечает значения, повторяющиеся min_hours часов подряд и дольше (застывший датчик)."""
    run_length = values.groupby(run_ids(values)).transform("size")
    return values.notna() & (run_length >= min_hours)


def check_clean_hourly(df: pd.DataFrame, columns: list[str], nan_columns=()) -> None:
    """Контракт для объединения (3.1): один час UTC — одна строка, без пропусков.

    Колонки из nan_columns могут быть пустыми (PM2.5 после длинных дыр).
    """
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
        if column not in nan_columns:
            assert df[column].isna().sum() == 0, f"в {column} есть пропуски"
    out_of_range = count_out_of_range(df, VALID_RANGES)
    assert sum(out_of_range.values()) == 0, f"значения вне диапазонов: {out_of_range}"


def check_clean_pm25(df: pd.DataFrame) -> None:
    """Контракт pm25_clean.csv для объединения (3.1)."""
    check_clean_hourly(df, PM25_COLUMNS, nan_columns=("pm25",))
    for flag in ["pm25_missing", "pm25_interpolated"]:
        assert pd.api.types.is_bool_dtype(df[flag]), f"{flag} не True/False"
    same_as_nan = (df["pm25_missing"] == df["pm25"].isna()).all()
    assert same_as_nan, "pm25_missing не совпадает с пустыми pm25"
    assert df.loc[df["pm25_interpolated"], "pm25"].notna().all(), "у заполненного часа нет значения"
    lengths = gap_lengths(df["pm25"])
    short = (lengths > 0) & (lengths <= MAX_INTERPOLATION_HOURS)
    assert not short.any(), "осталась дыра не длиннее 3 ч: её надо было заполнить"


def save_clean_csv(df: pd.DataFrame, path: str) -> None:
    """Записывает очищенную таблицу; файл перезаписывается при каждом запуске."""
    df.to_csv(path, index=False, date_format=PROCESSED_TIME_FORMAT)


def read_clean_csv(path: str) -> pd.DataFrame:
    """Читает очищенную таблицу так, как это делают 2.2 и 3.1."""
    df = pd.read_csv(path)
    df["time_utc"] = parse_utc_times(df["time_utc"], PROCESSED_TIME_FORMAT)
    return df
