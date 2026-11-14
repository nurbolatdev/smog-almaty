"""Таблицы для разведочного анализа и графиков (этап 4)."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

# подписи месяцев на графиках
MONTH_LABELS = {
    1: "янв",
    2: "фев",
    3: "мар",
    4: "апр",
    5: "май",
    6: "июн",
    7: "июл",
    8: "авг",
    9: "сен",
    10: "окт",
    11: "ноя",
    12: "дек",
}


def who_exceed_share_by_month(daily: pd.DataFrame) -> pd.DataFrame:
    """Доля суток с превышением нормы ВОЗ по месяцам; сутки без данных (<NA>) не считаются."""
    # count и sum пропускают <NA>: «нет данных» не становится «не превышено»
    grouped = daily.groupby("month")["who_exceed"]
    table = grouped.agg(days_known="count", days_exceed="sum").reset_index()
    table["share"] = table["days_exceed"] / table["days_known"]
    return table.sort_values("share", ascending=False).reset_index(drop=True)


def hour_month_table(hourly: pd.DataFrame, aggfunc: str = "mean") -> pd.DataFrame:
    """PM2.5 по часу местного времени (строки) и месяцу (колонки); пустые часы не участвуют."""
    return hourly.pivot_table(values="pm25", index="hour", columns="month", aggfunc=aggfunc)


def mean_with_ci(
    df: pd.DataFrame, value_column: str, by: list[str], confidence: float = 0.95
) -> pd.DataFrame:
    """Среднее по группам и t-интервал: mean ± t · std / √n (лекция 8)."""
    # count считает только непустые значения
    table = df.groupby(by)[value_column].agg(mean="mean", std="std", n="count").reset_index()
    t_value = stats.t.ppf((1 + confidence) / 2, table["n"] - 1)
    half_width = t_value * table["std"] / table["n"] ** 0.5
    # интервал предполагает независимые наблюдения; у соседних часов это не так,
    # поэтому настоящий интервал шире
    table["ci_low"] = table["mean"] - half_width
    table["ci_high"] = table["mean"] + half_width
    return table


def binned_medians(
    df: pd.DataFrame, x_column: str, y_column: str, n_bins: int = 10
) -> pd.DataFrame:
    """Медианы x и y в интервалах x с примерно равным числом строк (линия тренда)."""
    data = df[[x_column, y_column]].dropna()
    bins = pd.qcut(data[x_column], q=n_bins, duplicates="drop")
    grouped = data.groupby(bins, observed=True)
    table = grouped.agg(
        x_median=(x_column, "median"),
        y_median=(y_column, "median"),
        n=(y_column, "count"),
    )
    return table.reset_index(drop=True)


def spearman_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Корреляция Спирмена; строки с пропуском в любой колонке убираем."""
    # так все пары считаются на одних и тех же строках
    return df[columns].dropna().corr(method="spearman")


def heating_season_label(local_times: pd.Series) -> pd.Series:
    """Отопительный сезон по местному времени: '2020/21' — с осени 2020 по весну 2021."""
    # месяцы 1–6 относятся к сезону, начавшемуся в прошлом году;
    # часы вне отопительного сезона отбрасывает вызывающий код
    start_year = local_times.dt.year - (local_times.dt.month <= 6).astype(int)
    return start_year.astype(str) + "/" + (start_year + 1).astype(str).str[-2:]


def hour_profile_table(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    hour_column: str = "hour",
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Суточный профиль: час (строки) × группа (колонки); пустые значения не участвуют."""
    return df.pivot_table(
        values=value_column,
        index=hour_column,
        columns=group_column,
        aggfunc=aggfunc,
    )


def smooth_profile(profile: pd.Series, window: int = 3) -> pd.Series:
    """Скользящее среднее суточного профиля по кругу: после 23 ч идёт 0 ч."""
    assert len(profile) == 24 and profile.notna().all(), "нужны 24 часа без пропусков"
    assert window % 2 == 1 and window >= 1, "ширина окна должна быть нечётной"
    values = profile.to_numpy(dtype=float)
    half = window // 2
    total = np.zeros(24, dtype=float)
    # np.roll переставляет значения по кругу, поэтому 23 ч и 0 ч соседствуют
    for shift in range(-half, half + 1):
        total += np.roll(values, shift)
    return pd.Series(total / window, index=profile.index)


def profile_extremes(table: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Час и значение максимума и минимума сглаженного профиля для каждой колонки."""
    rows = []
    for column in table.columns:
        smooth = smooth_profile(table[column], window)
        rows.append(
            {
                "peak_hour": smooth.idxmax(),
                "peak": smooth.max(),
                "low_hour": smooth.idxmin(),
                "low": smooth.min(),
            }
        )
    return pd.DataFrame(rows, index=table.columns)


def profile_shift(
    reference: pd.Series, other: pd.Series, max_shift: int = 6
) -> pd.DataFrame:
    """Похожесть профилей при сдвиге по кругу; shift = +k — other запаздывает на k ч."""
    assert len(reference) == len(other) == 24, "нужны два профиля по 24 часа"
    ref_values = reference.to_numpy(dtype=float)
    other_values = other.to_numpy(dtype=float)
    shifts = list(range(-max_shift, max_shift + 1))
    corrs = []
    # положительный сдвиг переносит пик reference позже по часам
    for shift in shifts:
        corrs.append(float(np.corrcoef(other_values, np.roll(ref_values, shift))[0, 1]))
    return pd.DataFrame({"shift": shifts, "corr": corrs})


def lagged_correlations(
    df: pd.DataFrame,
    column: str,
    other_column: str,
    lags: Iterable[int],
    rows: pd.Series | None = None,
) -> pd.DataFrame:
    """ρ Спирмена между column и other_column со сдвигом lag ч; +k — column запаздывает на k ч."""
    steps = df["time_utc"].diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all(), (
        "нужна полная часовая сетка: сдвиг на строку = сдвиг на час"
    )
    if rows is None:
        rows = pd.Series(True, index=df.index)
    records = []
    # сдвигаем на всей сетке и только потом отбираем строки,
    # иначе на стыке сезонов соседними оказались бы часы с разрывом в полгода
    for lag in lags:
        pairs = pd.DataFrame({"x": df[column], "y": df[other_column].shift(lag)})[rows].dropna()
        rho = pairs["x"].corr(pairs["y"], method="spearman")
        records.append({"lag": lag, "rho": rho, "n": len(pairs)})
    return pd.DataFrame(records)
