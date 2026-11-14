"""Модели прогноза суточного PM2.5 (этап 6): таблица, разбиение по времени, метрики, модели."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def make_model_table(
    daily: pd.DataFrame, feature_columns: list[str], target: str = "pm25_mean"
) -> pd.DataFrame:
    """Собирает таблицу для модели без пропусков и приводит признаки к float."""
    cols = ["date", *feature_columns, target]
    # удаляем строки без цели или лага, а не заполняем: иначе модель училась бы на выдуманных днях
    table = daily[cols].dropna().reset_index(drop=True)
    for col in feature_columns:
        table[col] = table[col].astype(float)
    assert len(table) > 0, "после удаления пропусков не осталось строк"
    assert table["date"].is_monotonic_increasing and table["date"].is_unique, (
        "даты в таблице модели должны строго возрастать"
    )
    return table


def split_by_date(
    table: pd.DataFrame, test_start: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит таблицу по дате на обучение (< test_start) и тест (>= test_start)."""
    start = pd.Timestamp(test_start)
    # строгое '<' и '>=' гарантирует, что граничные сутки не попадут в обе части
    is_test = table["date"] >= start
    train = table.loc[~is_test].copy()
    test = table.loc[is_test].copy()
    assert not train.empty and not test.empty, "обучающая и тестовая части не должны быть пустыми"
    assert train["date"].max() < test["date"].min(), "обучение должно быть строго раньше теста"
    assert len(train) + len(test) == len(table), "потерялись строки при разбиении"
    return train, test


def naive_forecast(table: pd.DataFrame) -> pd.Series:
    """Наивный прогноз «завтра как сегодня» — вчерашний суточный PM2.5 (pm25_lag1)."""
    # значение pm25_lag1 в строке дня D равно среднему PM2.5 за сутки D-1
    pred = table["pm25_lag1"].astype(float).copy()
    assert pred.notna().all(), "в pm25_lag1 для наивного прогноза не должно быть пропусков"
    return pred


def month_mean_forecast(
    train: pd.DataFrame, test: pd.DataFrame, target: str = "pm25_mean"
) -> pd.Series:
    """Прогноз как среднее цели по календарному месяцу на обучающей части."""
    assert train[target].notna().all(), "в обучающей выборке не должно быть пропусков цели"
    means = train.groupby(train["date"].dt.month)[target].mean()
    pred = test["date"].dt.month.map(means).astype(float)
    assert pred.notna().all(), "в тесте есть месяц, которого не было в обучении"
    return pred


def regression_metrics(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> dict[str, float | int]:
    """Считает n, MAE, RMSE и R² между фактом и прогнозом."""
    assert len(y_true) == len(y_pred) and len(y_true) > 0, "нужны непустые массивы равной длины"
    assert not pd.isna(y_true).any() and not pd.isna(y_pred).any(), "в данных есть пропуски"
    # MAE = среднее |факт - прогноз|, основная метрика в µg/m³
    mae = float(mean_absolute_error(y_true, y_pred))
    # RMSE = sqrt(среднее (факт - прогноз)²), сильнее штрафует крупные промахи
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # R² = 1 - Σ(факт - прогноз)² / Σ(факт - среднее)², доля объяснённого разброса
    r2 = float(r2_score(y_true, y_pred))
    return {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "r2": r2}


def bootstrap_mae_diff(
    y_true: pd.Series | np.ndarray,
    pred_a: pd.Series | np.ndarray,
    pred_b: pd.Series | np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """Парный bootstrap разности ошибок (MAE_a - MAE_b) по одним и тем же суткам."""
    n = len(y_true)
    assert n > 0 and len(pred_a) == n and len(pred_b) == n, (
        "массивы должны быть непустыми и равными"
    )
    assert not (pd.isna(y_true).any() or pd.isna(pred_a).any() or pd.isna(pred_b).any()), (
        "в прогнозах или факте есть пропуски"
    )
    assert n_boot >= 1, "число bootstrap-выборок должно быть >= 1"
    y = np.asarray(y_true, dtype=float)
    err_a = np.abs(y - np.asarray(pred_a, dtype=float))
    err_b = np.abs(y - np.asarray(pred_b, dtype=float))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    # тянем одни и те же индексы суток для обеих моделей (парное сравнение на одних сутках)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = float(np.mean(err_a[idx]) - np.mean(err_b[idx]))
    return diffs
