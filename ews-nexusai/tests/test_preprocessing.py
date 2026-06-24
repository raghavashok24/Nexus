"""Sanity tests for src/data/preprocessing.py."""

import pandas as pd

from src.data.preprocessing import get_feature_groups, prepare_features_and_target


def _toy_panel(n_quarters: int = 10, banks_per_quarter: int = 5) -> pd.DataFrame:
    rows = []
    for q in range(1, n_quarters + 1):
        for b in range(banks_per_quarter):
            rows.append(
                {
                    "Quarter": q,
                    "Bank_Name": f"Bank_{b}",
                    "State": "CA",
                    "Region": "West",
                    "QuarterLabel": f"Q{q}",
                    "loans": 100.0 + b,
                    "tier1": 10.0,
                    "assets": 200.0,
                    "int_income": 5.0,
                    "int_expense": 2.0,
                    "LDR": 0.8,
                    "Tier1_to_Assets": 0.05,
                    "Net_Interest_Margin": 0.03,
                    "liquid_assets": 20.0,
                    "assets.1": 200.0,
                    "brokered_deposits": 10.0,
                    "deposits": 150.0,
                    "Liquid_to_Assets": 0.1,
                    "Brokered_to_Deposits": 0.06,
                    "Regional_InflationRate": 2.5,
                    "State_UnemploymentRate": 4.0,
                    "Total_Employees": 500,
                    "WARN_Layoffs": 0,
                    "Layoffs_per_1000": 0.0,
                    "Num_Layoff_Dates": 0,
                    "Avg_Notice_to_Layoff_Days": 0,
                    "Sentiment_of_Failure": 0.1,
                    "num_thresholds_exceeded": 0,
                    "x_was_imputed": 0,
                    "probabilistic_risk_score": 0.1 + 0.01 * q,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_features_and_target_splits_chronologically():
    df = _toy_panel(n_quarters=10)
    split = prepare_features_and_target(df, test_fraction=0.2)

    # Test set should only contain quarters strictly after the split point,
    # and all of them should be greater than every quarter in the train set.
    assert split.X_train["Quarter"].max() < split.X_test["Quarter"].min()
    assert len(split.X_train) + len(split.X_test) == len(df)


def test_prepare_features_and_target_drops_excluded_columns():
    df = _toy_panel(n_quarters=10)
    split = prepare_features_and_target(df)

    excluded = {"Bank_Name", "State", "Region", "QuarterLabel", "num_thresholds_exceeded", "x_was_imputed", "probabilistic_risk_score"}
    assert not excluded & set(split.X_train.columns)
    assert not excluded & set(split.X_test.columns)


def test_feature_groups_are_nested_supersets():
    groups = get_feature_groups()
    accounting = set(groups["Accounting Only"])
    acc_macro = set(groups["Accounting & Macro"])
    full = set(groups["Full Features"])

    assert accounting.issubset(acc_macro)
    assert acc_macro.issubset(full)
