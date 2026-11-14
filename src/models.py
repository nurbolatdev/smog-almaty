"""Модели прогноза суточного PM2.5 (этап 6): таблица, разбиение по времени, метрики, модели."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def make_linear_model() -> Pipeline:
    """Создаёт конвейер стандартизации признаков и линейной регрессии."""
    # StandardScaler внутри Pipeline учится только на данных fit (train), поэтому утечки нет;
    # на прогноз линейной регрессии масштаб не влияет, но делает коэффициенты сравнимыми
    return Pipeline([("scale", StandardScaler()), ("linreg", LinearRegression())])


def make_forest(
    n_estimators: int, max_depth: int, min_samples_leaf: int, seed: int = 42
) -> RandomForestRegressor:
    """Создаёт Random Forest с заданными параметрами и фиксированным seed."""
    assert n_estimators >= 1 and max_depth >= 1 and min_samples_leaf >= 1, (
        "параметры леса должны быть >= 1"
    )
    # деревья режут признаки по порогам, поэтому масштабирование лесу не нужно;
    # random_state фиксирует случайность обучения для воспроизводимости
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )


def linear_coefficients(pipeline: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    """Возвращает коэффициенты линейной модели (на 1 std и в исходных единицах)."""
    scaler = pipeline.named_steps["scale"]
    linreg = pipeline.named_steps["linreg"]
    assert hasattr(linreg, "coef_"), "модель должна быть обучена до чтения коэффициентов"
    assert len(feature_columns) == len(linreg.coef_), "число признаков не совпадает с моделью"
    coef_std = linreg.coef_
    # переводим коэффициент на 1 std в исходные единицы: coef_original = coef_std / scale_
    coef_original = coef_std / scaler.scale_
    table = pd.DataFrame(
        {"feature": feature_columns, "coef_std": coef_std, "coef_original": coef_original}
    )
    return table.sort_values("coef_std", key=np.abs, ascending=False).reset_index(drop=True)


def permutation_table(
    model,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Считает перестановочную важность признаков (рост MAE при перемешивании столбца)."""
    assert len(X) == len(y) and len(X) > 0, "X и y должны быть непустыми и равной длины"
    assert not X.isna().any().any() and not pd.isna(y).any(), "в данных есть пропуски"
    assert n_repeats >= 1, "число повторов должно быть >= 1"
    # при scoring='neg_mean_absolute_error' importances_mean даёт рост MAE со знаком +
    result = permutation_importance(
        model, X, y, scoring="neg_mean_absolute_error", n_repeats=n_repeats, random_state=seed
    )
    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "mae_increase": result.importances_mean,
            "mae_increase_std": result.importances_std,
        }
    )
    return table.sort_values("mae_increase", ascending=False).reset_index(drop=True)


def time_cv_table(
    table: pd.DataFrame,
    feature_columns: list[str],
    model,
    n_splits: int = 5,
    target: str = "pm25_mean",
) -> pd.DataFrame:
    """Оценивает модель и наивный прогноз на фолдах расширяющегося окна по времени."""
    assert table["date"].is_monotonic_increasing, "даты должны возрастать по времени"
    assert "pm25_lag1" in table.columns, "нужен столбец pm25_lag1 для наивного прогноза"
    # TimeSeriesSplit делит train на k+1 кусков: валидация всегда позже обучения
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(table), start=1):
        tr, val = table.iloc[train_idx], table.iloc[val_idx]
        # clone создаёт свежую необученную копию на каждый фолд; так проверяем только на train
        fitted = clone(model).fit(tr[feature_columns], tr[target])
        pred = fitted.predict(val[feature_columns])
        rows.append(
            {
                "fold": fold,
                "train_end": tr["date"].max(),
                "val_start": val["date"].min(),
                "val_end": val["date"].max(),
                "n_train": len(tr),
                "n_val": len(val),
                "mae_model": float(mean_absolute_error(val[target], pred)),
                "mae_naive": float(mean_absolute_error(val[target], naive_forecast(val))),
            }
        )
    return pd.DataFrame(rows)


def make_forest_grid(
    n_estimators_list: list[int], max_depths: list[int], min_leafs: list[int]
) -> list[dict[str, int]]:
    """Собирает список комбинаций гиперпараметров для перебора Random Forest."""
    assert n_estimators_list and max_depths and min_leafs, (
        "списки параметров не должны быть пустыми"
    )
    grid = []
    for n_est in n_estimators_list:
        for depth in max_depths:
            for leaf in min_leafs:
                grid.append(
                    {
                        "n_estimators": int(n_est),
                        "max_depth": int(depth),
                        "min_samples_leaf": int(leaf),
                    }
                )
    return grid


def forest_grid_cv(
    table: pd.DataFrame,
    feature_columns: list[str],
    grid: list[dict[str, int]],
    n_splits: int = 5,
    seed: int = 42,
    target: str = "pm25_mean",
) -> pd.DataFrame:
    """Перебирает сетку Random Forest в скользящей проверке по времени на train."""
    assert len(grid) > 0, "сетка параметров не должна быть пустой"
    rows = []
    for params in grid:
        forest = make_forest(
            int(params["n_estimators"]),
            int(params["max_depth"]),
            int(params["min_samples_leaf"]),
            seed=seed,
        )
        cv = time_cv_table(table, feature_columns, forest, n_splits=n_splits, target=target)
        rows.append(
            {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "cv_mae_mean": float(cv["mae_model"].mean()),
                "cv_mae_std": float(cv["mae_model"].std()),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_mae_mean").reset_index(drop=True)


def residuals_on_calendar(
    calendar_dates: pd.Series | pd.DatetimeIndex,
    dates: pd.Series | pd.DatetimeIndex,
    residuals: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Накладывает остатки на полный календарь дат, чтобы дыры стали NaN."""
    assert len(dates) == len(residuals), "число дат и остатков должно совпадать"
    calendar = pd.DataFrame({"date": pd.to_datetime(calendar_dates)}).sort_values("date")
    res_frame = pd.DataFrame(
        {"date": pd.to_datetime(dates), "residual": np.asarray(residuals, dtype=float)}
    )
    # без полного календаря shift/автокорреляция соединили бы сутки через дыру как соседние
    merged = calendar.merge(res_frame, on="date", how="left", validate="one_to_one")
    assert len(merged) == len(calendar), "merge не должен менять число календарных суток"
    return merged.reset_index(drop=True)
