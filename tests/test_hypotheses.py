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
