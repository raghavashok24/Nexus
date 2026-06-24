"""SHAP-based explainability for the tuned LightGBM model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Compute SHAP values for X_test using an explainer fit on X_train."""
    explainer = shap.Explainer(model, X_train)
    return explainer(X_test)


def mean_abs_shap_table(shap_values, feature_names) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value (global importance)."""
    return pd.DataFrame(
        {"Feature": feature_names, "MeanAbsSHAP": np.abs(shap_values.values).mean(axis=0)}
    ).sort_values(by="MeanAbsSHAP", ascending=False)
