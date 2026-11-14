"""Проверка гипотез (этап 5): сравнение групп, bootstrap, корреляция, χ²."""

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
