"""Regression tests confirming the original notebook's NameErrors are fixed.

The uploaded notebook had three cells that crashed with NameError because
they referenced variables (`ablation_df`, `predictions`) that were only
defined in a *different* cell, executed out of order. These tests build a
small synthetic dataset and run the equivalent module-based logic end to
end, with no dependency on global execution order, to confirm the bug
class can't recur.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.evaluation.ablation import run_ablation, pairwise_significance
from src.evaluation.precision_at_k import precision_at_k_benchmark
from src.evaluation.roc_and_correlation import binarize_target, auroc_with_bootstrap_ci


def _toy_split(n=120, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {
            "loans": rng.rand(n),
            "tier1": rng.rand(n),
            "assets": rng.rand(n),
            "int_income": rng.rand(n),
            "int_expense": rng.rand(n),
            "LDR": rng.rand(n),
            "Tier1_to_Assets": rng.rand(n),
            "Net_Interest_Margin": rng.rand(n),
            "liquid_assets": rng.rand(n),
            "assets.1": rng.rand(n),
            "brokered_deposits": rng.rand(n),
            "deposits": rng.rand(n),
            "Liquid_to_Assets": rng.rand(n),
            "Brokered_to_Deposits": rng.rand(n),
            "Regional_InflationRate": rng.rand(n),
            "State_UnemploymentRate": rng.rand(n),
            "Total_Employees": rng.rand(n),
            "WARN_Layoffs": rng.rand(n),
            "Layoffs_per_1000": rng.rand(n),
            "Num_Layoff_Dates": rng.rand(n),
            "Avg_Notice_to_Layoff_Days": rng.rand(n),
            "Sentiment_of_Failure": rng.rand(n),
        }
    )
    y = pd.Series(X["loans"] * 0.5 + X["tier1"] * 0.3 + rng.rand(n) * 0.05)
    split = n // 2
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def test_ablation_does_not_raise_nameerror_and_builds_dataframe():
    X_train, X_test, y_train, y_test = _toy_split()
    feature_groups = {
        "Accounting Only": ["loans", "tier1", "assets", "int_income", "int_expense", "LDR",
                             "Tier1_to_Assets", "Net_Interest_Margin", "liquid_assets",
                             "assets.1", "brokered_deposits", "deposits",
                             "Liquid_to_Assets", "Brokered_to_Deposits"],
        "Accounting & Macro": ["loans", "tier1", "Regional_InflationRate", "State_UnemploymentRate"],
        "Full Features": list(X_train.columns),
    }
    model = LinearRegression()

    result = run_ablation(model, X_train, X_test, y_train, y_test, feature_groups)

    # The dataframe that the original notebook crashed trying to build now
    # exists and is well-formed.
    assert list(result.summary_df["Feature Set"]) == result.ordered_labels
    assert {"R2", "CI Lower", "CI Upper"}.issubset(result.summary_df.columns)

    sig_df = pairwise_significance(result, y_test)
    assert len(sig_df) == 2


def test_precision_at_k_does_not_raise_nameerror_with_explicit_predictions():
    X_train, X_test, y_train, y_test = _toy_split()
    model = LinearRegression().fit(X_train, y_train)

    # `predictions` is built explicitly and passed in, instead of relying on
    # a same-named global from a previous cell.
    predictions = {"Linear": model.predict(X_test)}

    df, bootstrap_results = precision_at_k_benchmark(
        y_test, predictions, k_values=[5, 10], n_bootstrap=200
    )
    assert set(df["K"]) == {5, 10}
    assert "Linear" in bootstrap_results


def test_auroc_benchmark_runs_with_explicit_predictions():
    X_train, X_test, y_train, y_test = _toy_split()
    model = LinearRegression().fit(X_train, y_train)
    predictions = {"Linear": model.predict(X_test)}

    y_binary = binarize_target(y_test, threshold=y_test.median())
    df, boot_store = auroc_with_bootstrap_ci(y_binary, predictions, n_bootstrap=100)

    assert "Linear" in boot_store
    assert "AUROC" in df.columns
