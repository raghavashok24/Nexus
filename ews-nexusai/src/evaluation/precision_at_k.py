"""Precision@K: do the top-K predicted risk scores match the top-K true ones?

Fixes the original notebook's NameError by taking the `predictions` dict as
an explicit argument rather than relying on a variable defined in an
earlier, separately-run cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Fraction of the predicted top-K that overlaps with the true top-K."""
    pred_top_k = np.argsort(y_pred)[-k:][::-1]
    true_top_k = np.argsort(y_true)[-k:][::-1]
    return len(set(pred_top_k) & set(true_top_k)) / k


def bootstrap_precision_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, k: int, n_bootstrap: int = 1000, random_state: int = 42
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        scores[i] = precision_at_k(y_true[idx], y_pred[idx], k)
    return scores


def precision_at_k_benchmark(
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    k_values: list[int] = (5, 10, 20),
    n_bootstrap: int = 10000,
) -> tuple[pd.DataFrame, dict[str, dict[int, np.ndarray]]]:
    """Compute bootstrap Precision@K for every model and every K."""
    y_true = y_test.values
    bootstrap_results: dict[str, dict[int, np.ndarray]] = {
        model: {} for model in predictions
    }
    records = []

    for model_name, preds in predictions.items():
        for k in k_values:
            samples = bootstrap_precision_at_k(y_true, preds, k, n_bootstrap=n_bootstrap)
            bootstrap_results[model_name][k] = samples
            mean_score = samples.mean()
            lower_ci = np.percentile(samples, 2.5)
            upper_ci = np.percentile(samples, 97.5)
            records.append(
                {
                    "Model": model_name,
                    "K": k,
                    "Precision@K": mean_score,
                    "CI Lower": lower_ci,
                    "CI Upper": upper_ci,
                }
            )

    return pd.DataFrame(records), bootstrap_results


def wilcoxon_vs_reference(
    bootstrap_results: dict[str, dict[int, np.ndarray]],
    k_values: list[int],
    reference_model: str = "Tuned LightGBM",
) -> pd.DataFrame:
    """Wilcoxon signed-rank test of every other model against the reference model."""
    rows = []
    for k in k_values:
        ref_samples = bootstrap_results[reference_model][k]
        for model_name, by_k in bootstrap_results.items():
            if model_name == reference_model:
                continue
            stat, p_value = wilcoxon(ref_samples, by_k[k])
            rows.append(
                {
                    "K": k,
                    "Model": model_name,
                    "vs": reference_model,
                    "statistic": stat,
                    "p-value": p_value,
                    "Significant (a=0.05)": p_value < 0.05,
                }
            )
    return pd.DataFrame(rows)
