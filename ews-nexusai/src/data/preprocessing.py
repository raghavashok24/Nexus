"""Preprocessing: column hygiene, time-based train/test split, feature groups."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

TARGET_COL = "probabilistic_risk_score"

# Non-predictive metadata dropped before modeling.
NON_PREDICTIVE_COLS = ["Bank_Name", "State", "Region", "QuarterLabel"]

# Feature groups used by the ablation study (src/evaluation/ablation.py).
ACCOUNTING_FEATURES = [
    "loans", "tier1", "assets", "int_income", "int_expense", "LDR",
    "Tier1_to_Assets", "Net_Interest_Margin", "liquid_assets",
    "assets.1", "brokered_deposits", "deposits",
    "Liquid_to_Assets", "Brokered_to_Deposits",
]

MACRO_FEATURES = ["Regional_InflationRate", "State_UnemploymentRate"]

NON_ACCOUNTING_FEATURES = [
    "Total_Employees", "WARN_Layoffs", "Layoffs_per_1000",
    "Num_Layoff_Dates", "Avg_Notice_to_Layoff_Days", "Sentiment_of_Failure",
]

QUARTER_COL = ["Quarter"]


@dataclass
class SplitData:
    """Container for a time-based train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_fraction: float = 0.2,
) -> SplitData:
    """Clean columns and produce a time-based train/test split.

    Mirrors the original notebook's preprocessing cell:
      1. Drop ``*was_imputed*`` flag columns and ``num_thresholds_exceeded``
         from the model inputs (kept out to avoid leaking imputation/label
         construction signal into the features).
      2. Drop non-predictive metadata columns.
      3. Sort by ``Quarter`` and split chronologically — the last
         ``test_fraction`` of quarters become the held-out test set. This
         avoids leaking future information into training, which a random
         shuffle split would do for panel/time-series data.
    """
    flag_cols = [c for c in df.columns if "was_imputed" in c.lower()]
    exclude_from_model = ["num_thresholds_exceeded"] + flag_cols

    df = df.drop(columns=[c for c in NON_PREDICTIVE_COLS if c in df.columns])

    df_sorted = df.sort_values(by="Quarter").reset_index(drop=True)
    split_point = int(df_sorted["Quarter"].max() * (1 - test_fraction))

    train_df = df_sorted[df_sorted["Quarter"] <= split_point].copy()
    test_df = df_sorted[df_sorted["Quarter"] > split_point].copy()

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    X_train = train_df.drop(columns=[target_col] + exclude_from_model, errors="ignore")
    X_test = test_df.drop(columns=[target_col] + exclude_from_model, errors="ignore")

    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def make_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Return the cross-validation splitter used for Optuna tuning."""
    return TimeSeriesSplit(n_splits=n_splits)


def get_feature_groups() -> dict[str, list[str]]:
    """Return the named feature groups used by the ablation study."""
    return {
        "Accounting Only": ACCOUNTING_FEATURES,
        "Accounting & Macro": ACCOUNTING_FEATURES + MACRO_FEATURES,
        "Full Features": ACCOUNTING_FEATURES + MACRO_FEATURES + NON_ACCOUNTING_FEATURES,
    }
