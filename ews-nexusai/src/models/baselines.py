"""Untuned baseline models used for benchmarking against the tuned LightGBM."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def fit_baselines(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> dict[str, object]:
    """Fit Random Forest, Ridge, and XGBoost baseline regressors."""
    rf = RandomForestRegressor(random_state=random_state)
    ridge = Ridge()
    xgb = XGBRegressor(random_state=random_state)

    rf.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return {"Random Forest": rf, "Ridge Regression": ridge, "XGBoost": xgb}


def build_model_registry(tuned_lgb_model, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, object]:
    """Build the full {name: fitted_model} registry used across evaluation modules.

    The tuned LightGBM model is passed in pre-fitted (it's expensive to
    retrain); the other three are fit fresh here.
    """
    baselines = fit_baselines(X_train, y_train)
    return {"Tuned LightGBM": tuned_lgb_model, **baselines}


def predict_all(models: dict[str, object], X) -> dict[str, "np.ndarray"]:
    """Run `.predict(X)` for every model in the registry."""
    return {name: model.predict(X) for name, model in models.items()}
