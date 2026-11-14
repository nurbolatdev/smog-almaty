"""Таблицы для разведочного анализа и графиков (этап 4)."""

import pandas as pd

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
