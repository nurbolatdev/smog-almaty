"""Таблицы для разведочного анализа и графиков (этап 4)."""

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
