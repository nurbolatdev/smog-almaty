import numpy as np
import pandas as pd
import pytest

from src import hypotheses


def test_bootstrap_medians_reproducible_and_constant():
    constant = pd.Series([5.0] * 10)
    const_medians = hypotheses.bootstrap_medians(constant, n_boot=50, seed=42)
    assert len(const_medians) == 50
    assert (const_medians == 5.0).all()

    values = pd.Series(np.arange(1.0, 21.0))
    same_1 = hypotheses.bootstrap_medians(values, n_boot=100, seed=7)
    same_2 = hypotheses.bootstrap_medians(values, n_boot=100, seed=7)
    other = hypotheses.bootstrap_medians(values, n_boot=100, seed=8)
    assert np.array_equal(same_1, same_2)
    assert not np.array_equal(same_1, other)


def test_percentile_ci_middle_95():
    low, high = hypotheses.percentile_ci(np.arange(101))
    assert (low, high) == pytest.approx((2.5, 97.5))


def test_compare_groups_shifted_groups():
    b = pd.Series(np.arange(1.0, 21.0))
    a = b + 10.0

    result = hypotheses.compare_groups(a, b, n_boot=500, seed=42)

    assert result["n_a"] == 20
    assert result["n_b"] == 20
    assert result["median_diff"] == 10.0
    assert result["welch_p"] < 0.05
    assert result["mw_p"] < 0.05
    assert result["ci_low"] <= 10.0 <= result["ci_high"]


def test_compare_groups_identical_groups():
    values = pd.Series(np.arange(1.0, 21.0))

    result = hypotheses.compare_groups(values, values, n_boot=500, seed=42)

    assert result["mw_p"] == pytest.approx(1.0)
    assert result["welch_p"] == pytest.approx(1.0)
    assert result["prob_superiority"] == 0.5
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_compare_groups_separated_groups_superiority():
    b = pd.Series(np.arange(1.0, 21.0))
    a = b + 100.0

    result = hypotheses.compare_groups(a, b, n_boot=200, seed=42)

    assert result["prob_superiority"] == 1.0


def test_compare_groups_rejects_nan():
    b = pd.Series(np.arange(1.0, 21.0))
    a = b.copy()
    a.iloc[0] = np.nan

    with pytest.raises(AssertionError):
        hypotheses.compare_groups(a, b, n_boot=100, seed=42)


def test_format_p():
    assert hypotheses.format_p(0.0004) == "< 0.001"
    assert hypotheses.format_p(0.0123) == "0.012"
    assert hypotheses.format_p(0.5) == "0.500"


def test_spearman_with_ci_monotonic_and_drops_nan():
    x = np.arange(1.0, 31.0)
    y = x**3
    y[-1] = np.nan
    df = pd.DataFrame({"x": x, "y": y})

    result = hypotheses.spearman_with_ci(df, "x", "y", n_boot=200, seed=42)

    assert result["n"] == 29
    assert result["rho"] == pytest.approx(1.0)
    assert result["ci_low"] == pytest.approx(1.0)


def test_chi2_table_expected_counts():
    rows = [("a", True)] * 10 + [("a", False)] * 20
    rows += [("b", True)] * 30 + [("b", False)] * 40
    rows += [("c", True)] * 20 + [("c", False)] * 30
    df = pd.DataFrame(rows, columns=["group", "flag"])

    result = hypotheses.chi2_table(df, "group", "flag")

    assert result["dof"] == 2
    assert result["n"] == 150
    expected = result["expected"]
    # вручную: итог строки * итог колонки / n (True всего 60, False всего 90)
    assert expected.loc["a", True] == pytest.approx(30 * 60 / 150)
    assert expected.loc["b", False] == pytest.approx(70 * 90 / 150)
    assert expected.loc["c", True] == pytest.approx(50 * 60 / 150)
    assert result["min_expected"] == pytest.approx(expected.to_numpy().min())


def test_chi2_table_effect_size_extremes():
    equal_rows = (
        [("a", True)] * 5
        + [("a", False)] * 5
        + [("b", True)] * 10
        + [("b", False)] * 10
        + [("c", True)] * 15
        + [("c", False)] * 15
    )
    equal_res = hypotheses.chi2_table(pd.DataFrame(equal_rows, columns=["g", "f"]), "g", "f")
    assert equal_res["chi2"] == pytest.approx(0.0)
    assert equal_res["cramers_v"] == pytest.approx(0.0)
    assert equal_res["p"] == pytest.approx(1.0)

    split_rows = [("a", True)] * 10 + [("b", False)] * 10 + [("c", False)] * 10
    split_res = hypotheses.chi2_table(pd.DataFrame(split_rows, columns=["g", "f"]), "g", "f")
    assert split_res["cramers_v"] == pytest.approx(1.0)

    with_na = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "f": pd.array([True, pd.NA, False, True], dtype="boolean"),
        }
    )
    with pytest.raises(AssertionError):
        hypotheses.chi2_table(with_na, "g", "f")


def test_autocorrelation_by_lag_alternating_series():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "pm25_mean": [1.0, 10.0] * 10,
        }
    )

    res = hypotheses.autocorrelation_by_lag(daily, "pm25_mean", [1, 2])

    lag1 = res[res["lag"] == 1].iloc[0]
    lag2 = res[res["lag"] == 2].iloc[0]
    assert lag1["rho"] == pytest.approx(-1.0)
    assert lag1["n"] == 19
    assert lag2["rho"] == pytest.approx(1.0)

    first_ten = pd.Series([True] * 10 + [False] * 10, index=daily.index)
    res_ten = hypotheses.autocorrelation_by_lag(daily, "pm25_mean", [1], rows=first_ten)
    assert res_ten.iloc[0]["n"] == 9

    with_gap = daily.drop(index=5).reset_index(drop=True)
    with pytest.raises(AssertionError):
        hypotheses.autocorrelation_by_lag(with_gap, "pm25_mean", [1])


def test_thin_days_positions_and_checks():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": np.arange(10),
        }
    )

    thinned = hypotheses.thin_days(daily, step=3, offset=1)
    assert thinned["value"].tolist() == [1, 4, 7]

    with pytest.raises(AssertionError):
        hypotheses.thin_days(daily, step=3, offset=3)

    with_gap = daily.drop(index=4).reset_index(drop=True)
    with pytest.raises(AssertionError):
        hypotheses.thin_days(with_gap, step=3, offset=0)
