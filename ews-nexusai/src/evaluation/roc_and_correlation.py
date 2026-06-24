"""ROC/AUC and Spearman rank-correlation benchmarking across models.

Fixes the original notebook's NameError (cells referenced a bare
`predictions` dict that was never assigned before use): here, the
predictions dict is always an explicit function argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from src.evaluation.bootstrap import spearman_bootstrap_ci


def binarize_target(y: pd.Series, threshold: float = 0.5) -> np.ndarray:
    """Binarize the continuous risk score for ROC/AUC evaluation."""
    return (y >= threshold).astype(int).values


def auroc_with_bootstrap_ci(
    y_true_binary: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute AUROC with a bootstrap 95% CI for each model's predictions."""
    rng = np.random.RandomState(random_state)
    n = len(y_true_binary)
    rows = []
    boot_store: dict[str, np.ndarray] = {}

    for name, preds in predictions.items():
        auc = roc_auc_score(y_true_binary, preds)
        boots = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            if len(np.unique(y_true_binary[idx])) < 2:
                boots[i] = np.nan
                continue
            boots[i] = roc_auc_score(y_true_binary[idx], preds[idx])
        boot_store[name] = boots
        valid = boots[~np.isnan(boots)]
        if len(valid) == 0:
            ci_lower, ci_upper = np.nan, np.nan
        else:
            ci_lower, ci_upper = np.percentile(valid, 2.5), np.percentile(valid, 97.5)
        rows.append(
            {
                "Model": name,
                "AUROC": auc,
                "CI Lower": ci_lower,
                "CI Upper": ci_upper,
            }
        )

    return pd.DataFrame(rows).sort_values("AUROC", ascending=False), boot_store


def pairwise_auroc_tests(boot_store: dict[str, np.ndarray]) -> pd.DataFrame:
    """Bootstrap pairwise AUROC comparison across all model pairs."""
    from itertools import combinations

    rows = []
    for m1, m2 in combinations(boot_store.keys(), 2):
        b1, b2 = boot_store[m1], boot_store[m2]
        mask = ~(np.isnan(b1) | np.isnan(b2))
        diffs = b1[mask] - b2[mask]
        if len(diffs) == 0:
            rows.append({"Model 1": m1, "Model 2": m2, "Mean Delta AUC": np.nan, "p-value": np.nan})
            continue
        mean_diff = diffs.mean()
        p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        rows.append({"Model 1": m1, "Model 2": m2, "Mean Delta AUC": mean_diff, "p-value": p_value})
    return pd.DataFrame(rows)


def spearman_benchmark(
    y_test: pd.Series, predictions: dict[str, np.ndarray], n_bootstrap: int = 1000
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute bootstrap Spearman correlation for each model's predictions."""
    rows = []
    boot_store: dict[str, np.ndarray] = {}
    for name, preds in predictions.items():
        mean_corr, ci_lower, ci_upper, boots = spearman_bootstrap_ci(
            y_test.values, preds, n_bootstrap=n_bootstrap
        )
        boot_store[name] = boots
        rows.append({"Model": name, "Mean": mean_corr, "CI Lower": ci_lower, "CI Upper": ci_upper})

    df = pd.DataFrame(rows).sort_values("Mean", ascending=False)
    return df, boot_store


def wilcoxon_compare(boot_store: dict[str, np.ndarray], model_a: str, model_b: str):
    """Wilcoxon signed-rank test between two models' bootstrap sample arrays."""
    return wilcoxon(boot_store[model_a], boot_store[model_b])
