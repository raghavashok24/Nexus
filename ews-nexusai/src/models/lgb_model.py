"""LightGBM baseline, Optuna hyperparameter tuning, and feature importance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


@dataclass
class TunedModelResult:
    model: LGBMRegressor
    best_params: dict
    cv_mean_mse: float
    cv_ci: tuple[float, float]


def fit_baseline_lgb(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> LGBMRegressor:
    """Fit an untuned LightGBM regressor as the baseline model."""
    model = LGBMRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def _objective_factory(X_train: pd.DataFrame, y_train: pd.Series, cv):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", -1, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
            "force_col_wise": True,
            "n_jobs": -1,
        }
        model = LGBMRegressor(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
        )
        return -scores.mean()

    return objective


def tune_lgb_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv,
    n_trials: int = 40,
    seed: int = 42,
) -> TunedModelResult:
    """Run Optuna hyperparameter search and fit the best LightGBM model.

    Reports CV mean MSE with a 95% interval (2.5th/97.5th percentile of
    per-fold scores) and refits on the full training set with the best
    found hyperparameters, matching the original notebook's tuning cell.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective_factory(X_train, y_train, cv), n_trials=n_trials)

    best_params = dict(study.best_params)
    best_params.update({"random_state": seed, "force_col_wise": True, "n_jobs": -1})

    model = LGBMRegressor(**best_params)
    model.fit(X_train, y_train)

    cv_scores = -cross_val_score(
        LGBMRegressor(**best_params), X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
    )
    ci_lower, ci_upper = np.percentile(cv_scores, [2.5, 97.5])

    return TunedModelResult(
        model=model,
        best_params=best_params,
        cv_mean_mse=float(cv_scores.mean()),
        cv_ci=(float(ci_lower), float(ci_upper)),
    )


def evaluate(model, X, y) -> dict[str, float]:
    """Compute MSE and R² for a fitted regressor on a given split."""
    preds = model.predict(X)
    return {
        "mse": float(mean_squared_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }


def feature_importance_table(model: LGBMRegressor, feature_names) -> pd.DataFrame:
    """Return a sorted dataframe of LightGBM split-based feature importances."""
    return pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
