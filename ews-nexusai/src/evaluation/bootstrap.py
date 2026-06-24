"""Bootstrap confidence intervals and significance tests.

These are the generic bootstrap routines used throughout the evaluation
pipeline (ablation study, Spearman correlation, precision@K). Centralizing
them here also fixes the main bug in the original notebook: cells that
referenced ``ablation_df`` or ``predictions`` before either variable was
defined now get those values passed in explicitly as function arguments,
so there's no reliance on cell execution order.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    ci: int = 95,
    random_state: int = 42,
) -> tuple[float, float, float, np.ndarray]:
    """Bootstrap a confidence interval for an arbitrary metric function."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        scores[i] = metric_func(y_true[idx], y_pred[idx])

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return float(scores.mean()), float(lower), float(upper), scores


def bootstrap_metric_diff_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    ci: int = 95,
    random_state: int = 42,
) -> tuple[float, float, float, float]:
    """Bootstrap test for whether metric(model1) differs from metric(model2)."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    n = len(y_true)

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        diffs[i] = metric_func(y_true[idx], y_pred1[idx]) - metric_func(y_true[idx], y_pred2[idx])

    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(diffs.mean()), float(lower), float(upper), float(p_value)


def spearman_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: int = 95,
    random_state: int = 42,
) -> tuple[float, float, float, np.ndarray]:
    """Bootstrap CI for the Spearman rank correlation between y_true and y_pred."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    boot_corrs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        corr, _ = spearmanr(y_true[idx], y_pred[idx])
        boot_corrs[i] = corr

    lower = np.percentile(boot_corrs, (100 - ci) / 2)
    upper = np.percentile(boot_corrs, 100 - (100 - ci) / 2)
    return float(boot_corrs.mean()), float(lower), float(upper), boot_corrs
