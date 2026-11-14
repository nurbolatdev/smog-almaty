"""Проверка гипотез (этап 5): сравнение групп, bootstrap, корреляция, χ²."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def _to_array(values: pd.Series | np.ndarray) -> np.ndarray:
    """Проверяет группу (>= 2 значений, без пропусков) и возвращает float-массив."""
    assert len(values) >= 2 and not pd.isna(values).any(), (
        "в группе должно быть >= 2 значений без пропусков"
    )
    return np.asarray(values, dtype=float)


def bootstrap_medians(
    values: pd.Series | np.ndarray, n_boot: int = 5000, seed: int = 42
) -> np.ndarray:
    """Медианы n_boot bootstrap-выборок с возвращением (лекция 8)."""
    data = _to_array(values)
    assert n_boot >= 1, "число bootstrap-выборок должно быть >= 1"
    rng = np.random.default_rng(seed)
    medians = np.empty(n_boot, dtype=float)
    # на каждом шаге тянем выборку того же размера с возвращением и считаем медиану
    for i in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        medians[i] = np.median(sample)
    return medians


def bootstrap_median_diffs(
    a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, n_boot: int = 5000, seed: int = 42
) -> np.ndarray:
    """Bootstrap-разницы медиан (a - b) при независимой выборке двух групп."""
    # разные seed, чтобы выборки двух групп тянуть независимо
    # (при одном seed и одинаковом размере групп индексы совпали бы)
    return bootstrap_medians(a, n_boot, seed) - bootstrap_medians(b, n_boot, seed + 1)


def percentile_ci(
    values: pd.Series | np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Перцентильный доверительный интервал (средние confidence * 100 % оценок)."""
    assert len(values) >= 1 and not pd.isna(values).any(), "нужны значения без пропусков"
    assert 0 < confidence < 1, "уровень доверия должен быть строго между 0 и 1"
    tail = (100.0 - 100.0 * confidence) / 2.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def compare_groups(
    a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, n_boot: int = 5000, seed: int = 42
) -> dict[str, float | int]:
    """Сравнивает группы a и b: Welch t-test, Mann-Whitney U и bootstrap CI разницы медиан."""
    arr_a = _to_array(a)
    arr_b = _to_array(b)
    # equal_var=False: t-тест Уэлча не требует равенства дисперсий двух групп
    welch = stats.ttest_ind(arr_a, arr_b, equal_var=False)
    # alternative="two-sided": проверяем сдвиг в любую сторону (консервативнее одностороннего)
    mw = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    ci_low, ci_high = percentile_ci(bootstrap_median_diffs(arr_a, arr_b, n_boot, seed))
    median_a = float(np.median(arr_a))
    median_b = float(np.median(arr_b))
    # prob_superiority: доля пар (из a, из b), где значение в группе a больше (ничьи пополам)
    return {
        "n_a": int(len(arr_a)),
        "n_b": int(len(arr_b)),
        "mean_a": float(np.mean(arr_a)),
        "mean_b": float(np.mean(arr_b)),
        "median_a": median_a,
        "median_b": median_b,
        "welch_t": float(welch.statistic),
        "welch_p": float(welch.pvalue),
        "mw_u": float(mw.statistic),
        "mw_p": float(mw.pvalue),
        "prob_superiority": float(mw.statistic / (len(arr_a) * len(arr_b))),
        "median_diff": float(median_a - median_b),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def format_p(p: float) -> str:
    """Форматирует p-value: '< 0.001' для очень малых значений, иначе 3 знака."""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def spearman_with_ci(
    df: pd.DataFrame, x_column: str, y_column: str, n_boot: int = 5000, seed: int = 42
) -> dict[str, float | int]:
    """Корреляция Спирмена и bootstrap 95 % CI по парам строк."""
    pairs = df[[x_column, y_column]].dropna()
    n = len(pairs)
    assert n >= 2 and n_boot >= 1, "нужно >= 2 пар без пропусков и n_boot >= 1"
    x = pairs[x_column].to_numpy(dtype=float)
    y = pairs[y_column].to_numpy(dtype=float)
    res = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    rhos = np.empty(n_boot, dtype=float)
    # тянем пары строк целиком, а не колонки по отдельности — связь живёт в парах
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rhos[i] = stats.spearmanr(x[idx], y[idx]).statistic
    ci_low, ci_high = percentile_ci(rhos)
    return {
        "n": int(n),
        "rho": float(res.statistic),
        "p": float(res.pvalue),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def chi2_table(df: pd.DataFrame, row_column: str, col_column: str) -> dict[str, object]:
    """Таблица сопряжённости, критерий χ² Пирсона, ожидаемые частоты и Cramér's V."""
    assert df[row_column].notna().all() and df[col_column].notna().all(), (
        "в колонках для χ² не должно быть пропусков"
    )
    table = pd.crosstab(df[row_column], df[col_column])
    assert min(table.shape) >= 2, "таблица сопряжённости должна быть не меньше 2×2"
    # поправку Йейтса scipy применяет только к таблице 2×2 (dof = 1); у нас 4×2
    chi2, p, dof, expected_arr = stats.chi2_contingency(table)
    expected = pd.DataFrame(expected_arr, index=table.index, columns=table.columns)
    n = int(len(df))
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {
        "table": table,
        "expected": expected,
        "min_expected": float(expected_arr.min()),
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "n": n,
        "cramers_v": cramers_v,
    }


def _check_consecutive_dates(daily: pd.DataFrame) -> None:
    """Проверяет, что в таблице даты идут подряд с шагом ровно 1 сутки."""
    assert daily["date"].notna().all(), "в date есть пропуски"
    steps = daily["date"].diff().dropna()
    assert (steps == pd.Timedelta(days=1)).all(), "даты должны идти подряд с шагом 1 сутки"


def autocorrelation_by_lag(
    daily: pd.DataFrame,
    column: str,
    lags: Iterable[int],
    rows: pd.Series | None = None,
) -> pd.DataFrame:
    """Автокорреляция Спирмена между column и сдвигом на lag суток."""
    _check_consecutive_dates(daily)
    if rows is None:
        rows = pd.Series(True, index=daily.index)
    records = []
    # пара берётся, только если оба дня (t и t - lag) входят в выбранную группу
    for lag in lags:
        both = rows & rows.shift(lag, fill_value=False)
        pairs = pd.DataFrame({"x": daily[column], "y": daily[column].shift(lag)})[both].dropna()
        rho = float(pairs["x"].corr(pairs["y"], method="spearman"))
        records.append({"lag": int(lag), "rho": rho, "n": int(len(pairs))})
    return pd.DataFrame(records)


def thin_days(daily: pd.DataFrame, step: int, offset: int = 0) -> pd.DataFrame:
    """Берёт каждые step-е календарные сутки, начиная со сдвига offset."""
    _check_consecutive_dates(daily)
    assert step >= 1 and 0 <= offset < step, "должно выполняться 0 <= offset < step"
    # прореживаем календарные сутки, а не строки со значением —
    # иначе после пропуска 2022 года соседними оказались бы дни с разрывом в месяцы
    return daily.iloc[offset::step].copy()
