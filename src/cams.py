"""Сравнение модели CAMS с измерениями станции (гипотеза H4, этап 5)."""

import numpy as np
import pandas as pd
from scipy import stats

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


def agreement_metrics(
    df: pd.DataFrame,
    station_column: str = "pm25",
    model_column: str = "cams_pm25",
) -> dict[str, float | int]:
    """Метрики согласия модели со станцией: bias (µg/m³ и %), MAE, Пирсон и Спирмен."""
    assert {station_column, model_column}.issubset(df.columns), (
        "в таблице не хватает колонок для сравнения"
    )
    pairs = df[[station_column, model_column]].dropna()
    assert len(pairs) >= 2, "нужно >= 2 пар без пропусков"
    station = pairs[station_column].to_numpy(dtype=float)
    model = pairs[model_column].to_numpy(dtype=float)
    diff = model - station
    # bias = среднее(CAMS − станция): знак < 0 означает, что CAMS занижает;
    # MAE = среднее(|CAMS − станция|) в µg/m³
    station_mean = float(np.mean(station))
    assert station_mean != 0, "среднее станции равно нулю — bias_pct не определён"
    bias = float(np.mean(diff))
    # Пирсон (r) — линейная связь из постановки, Спирмен (ρ) — устойчив к пикам PM2.5
    return {
        "n": int(len(pairs)),
        "station_mean": station_mean,
        "cams_mean": float(np.mean(model)),
        "bias": bias,
        "bias_pct": float(bias / station_mean * 100.0),
        "mae": float(np.mean(np.abs(diff))),
        "pearson_r": float(np.corrcoef(station, model)[0, 1]),
        "spearman_rho": float(stats.spearmanr(station, model).statistic),
    }


def bootstrap_means(
    values: pd.Series | np.ndarray, n_boot: int = 5000, seed: int = 42
) -> np.ndarray:
    """Средние n_boot bootstrap-выборок с возвращением (лекция 8)."""
    assert len(values) >= 2 and not pd.isna(values).any(), (
        "в выборке должно быть >= 2 значений без пропусков"
    )
    assert n_boot >= 1, "число bootstrap-выборок должно быть >= 1"
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    # на каждом шаге тянем выборку с возвращением того же размера и считаем среднее
    for i in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        means[i] = np.mean(sample)
    return means
