"""Feature-group ablation study: Accounting Only -> + Macro -> Full Features.

This module fixes the original notebook's NameError in the early ablation
cell (`ablation_df` was referenced before assignment): here, the dataframe
used for plotting is always built from the same `ablation_results` dict that
the fits produce, in a single function, before any plotting code runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score

from src.evaluation.bootstrap import bootstrap_metric_ci, bootstrap_metric_diff_test

ORDERED_LABELS = ["Accounting Only", "Accounting & Macro", "Full Features"]


@dataclass
class AblationResult:
    ordered_labels: list[str]
    r2_by_label: dict[str, float]
    predictions_by_label: dict[str, np.ndarray]
    ci_by_label: dict[str, tuple[float, float, float]]
    summary_df: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.summary_df = pd.DataFrame(
            {
                "Feature Set": self.ordered_labels,
                "R2": [self.r2_by_label[l] for l in self.ordered_labels],
                "CI Lower": [self.ci_by_label[l][1] for l in self.ordered_labels],
                "CI Upper": [self.ci_by_label[l][2] for l in self.ordered_labels],
            }
        )


def run_ablation(
    tuned_model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_groups: dict[str, list[str]],
    ordered_labels: list[str] = ORDERED_LABELS,
    n_bootstrap: int = 1000,
) -> AblationResult:
    """Fit a cloned tuned model on each feature group and bootstrap R² CIs."""
    r2_by_label: dict[str, float] = {}
    predictions_by_label: dict[str, np.ndarray] = {}
    ci_by_label: dict[str, tuple[float, float, float]] = {}

    for label in ordered_labels:
        cols = feature_groups[label]
        model = clone(tuned_model)
        model.fit(X_train[cols], y_train)
        preds = model.predict(X_test[cols])

        r2_by_label[label] = r2_score(y_test, preds)
        predictions_by_label[label] = preds

        mean, lower, upper, _ = bootstrap_metric_ci(
            y_test.values, preds, r2_score, n_bootstrap=n_bootstrap
        )
        ci_by_label[label] = (mean, lower, upper)

    return AblationResult(
        ordered_labels=ordered_labels,
        r2_by_label=r2_by_label,
        predictions_by_label=predictions_by_label,
        ci_by_label=ci_by_label,
    )


def pairwise_significance(
    result: AblationResult,
    y_test: pd.Series,
    pairs: list[tuple[str, str]] | None = None,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Bootstrap significance test for R² improvement between feature-group pairs."""
    if pairs is None:
        pairs = [
            ("Accounting & Macro", "Accounting Only"),
            ("Full Features", "Accounting & Macro"),
        ]

    rows = []
    for higher, lower in pairs:
        mean_diff, ci_lower, ci_upper, p_value = bootstrap_metric_diff_test(
            y_test.values,
            result.predictions_by_label[higher],
            result.predictions_by_label[lower],
            r2_score,
            n_bootstrap=n_bootstrap,
        )
        rows.append(
            {
                "Comparison": f"{higher} vs {lower}",
                "Mean R2 Diff": mean_diff,
                "CI Lower": ci_lower,
                "CI Upper": ci_upper,
                "p-value": p_value,
                "Significant (a=0.05)": p_value < 0.05,
            }
        )
    return pd.DataFrame(rows)
