"""
XGBoost goal predictor with Optuna hyperparameter optimisation.

Design
------
Two independent XGBoost regressors are trained:
  - model_home : predicts λ_home  (expected home goals)
  - model_away : predicts λ_away  (expected away goals)

Both use objective='count:poisson' so the output is already on the
expected-count (λ) scale — no post-processing sigmoid or softmax needed.

A third XGBoost classifier predicts P(result ∈ {H, D, A}) directly and is
used to cross-validate the Poisson-derived match probabilities.

All training is done with temporal sample weights (exponential decay) so
recent matches are emphasised without discarding old data entirely.

Hyperparameter search
---------------------
Optuna minimises mean Poisson deviance on a hold-out fold (the most recent
20% of training data, preserving temporal order).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from src.config import (
    XGBOOST_MODEL_DIR,
    XGBOOST_OPTUNA_TRIALS,
    XGBOOST_RANDOM_SEED,
)
from src.features.pipeline import FEATURE_COLUMNS

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_MODEL_HOME_FILE = "model_home.ubj"
_MODEL_AWAY_FILE = "model_away.ubj"
_MODEL_RESULT_FILE = "model_result.ubj"


def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance (lower is better)."""
    eps = 1e-8
    pred = np.clip(y_pred, eps, None)
    return float(np.mean(2 * (y_true * np.log((y_true + eps) / pred) - (y_true - pred))))


def _build_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    w_train: np.ndarray,
    n_trials: int,
    seed: int,
) -> dict:
    """Run Optuna to find best XGBoost hyperparameters for Poisson regression."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "count:poisson",
            "max_delta_step": trial.suggest_float("max_delta_step", 0.5, 1.5),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        y_pred = np.clip(model.predict(X_val), 0.0, None)
        return _poisson_deviance(y_val, y_pred)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna best value: %.4f  params: %s", study.best_value, study.best_params)
    return study.best_params


class XGBoostPredictor:
    """
    Dual Poisson XGBoost predictor for home and away goals.

    Predictions are λ values (expected goals) compatible with the Poisson
    matrix and ensemble combination.
    """

    def __init__(self) -> None:
        self.model_home: Optional[xgb.XGBRegressor] = None
        self.model_away: Optional[xgb.XGBRegressor] = None
        self.model_result: Optional[xgb.XGBClassifier] = None
        self.feature_cols: list[str] = FEATURE_COLUMNS

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        optimize: bool = True,
        n_trials: int = XGBOOST_OPTUNA_TRIALS,
    ) -> "XGBoostPredictor":
        """
        Train home/away Poisson regressors and a result classifier.

        *df* must already contain all feature columns and target columns
        (home_score, away_score, result) as well as sample_weight.
        """
        avail = [c for c in self.feature_cols if c in df.columns]
        X = df[avail].fillna(0.0).values
        y_home = df["home_score"].values.astype(float)
        y_away = df["away_score"].values.astype(float)
        y_result = (df["result"].map({1: 0, 0: 1, -1: 2}).fillna(1)).values.astype(int)

        weights = df["sample_weight"].values if "sample_weight" in df.columns else np.ones(len(df))

        # Temporal train/validation split (last 20% = hold-out)
        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        yh_tr, yh_val = y_home[:split], y_home[split:]
        ya_tr, ya_val = y_away[:split], y_away[split:]
        w_tr = weights[:split]

        # ── Home goals ────────────────────────────────────────────────────────
        logger.info("Optimising XGBoost for home goals …")
        if optimize:
            best_home = _build_study(X_tr, yh_tr, X_val, yh_val, w_tr, n_trials, XGBOOST_RANDOM_SEED)
        else:
            best_home = {}

        self.model_home = xgb.XGBRegressor(
            objective="count:poisson",
            max_delta_step=best_home.get("max_delta_step", 0.7),
            n_estimators=best_home.get("n_estimators", 500),
            learning_rate=best_home.get("learning_rate", 0.05),
            max_depth=best_home.get("max_depth", 5),
            min_child_weight=best_home.get("min_child_weight", 3),
            subsample=best_home.get("subsample", 0.8),
            colsample_bytree=best_home.get("colsample_bytree", 0.8),
            reg_alpha=best_home.get("reg_alpha", 0.1),
            reg_lambda=best_home.get("reg_lambda", 1.0),
            random_state=XGBOOST_RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
        self.model_home.fit(X, y_home, sample_weight=weights, verbose=False)
        h_pred = np.clip(self.model_home.predict(X_val), 0.0, None)
        logger.info("Home RMSE (val): %.4f", np.sqrt(mean_squared_error(yh_val, h_pred)))

        # ── Away goals ────────────────────────────────────────────────────────
        logger.info("Optimising XGBoost for away goals …")
        if optimize:
            best_away = _build_study(X_tr, ya_tr, X_val, ya_val, w_tr, n_trials, XGBOOST_RANDOM_SEED + 1)
        else:
            best_away = {}

        self.model_away = xgb.XGBRegressor(
            objective="count:poisson",
            max_delta_step=best_away.get("max_delta_step", 0.7),
            n_estimators=best_away.get("n_estimators", 500),
            learning_rate=best_away.get("learning_rate", 0.05),
            max_depth=best_away.get("max_depth", 5),
            min_child_weight=best_away.get("min_child_weight", 3),
            subsample=best_away.get("subsample", 0.8),
            colsample_bytree=best_away.get("colsample_bytree", 0.8),
            reg_alpha=best_away.get("reg_alpha", 0.1),
            reg_lambda=best_away.get("reg_lambda", 1.0),
            random_state=XGBOOST_RANDOM_SEED + 1,
            n_jobs=-1,
            verbosity=0,
        )
        self.model_away.fit(X, y_away, sample_weight=weights, verbose=False)
        a_pred = np.clip(self.model_away.predict(X_val), 0.0, None)
        logger.info("Away RMSE (val): %.4f", np.sqrt(mean_squared_error(ya_val, a_pred)))

        # ── Result classifier ─────────────────────────────────────────────────
        logger.info("Training result classifier …")
        self.model_result = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=XGBOOST_RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
        self.model_result.fit(X, y_result, sample_weight=weights, verbose=False)

        self.feature_cols = avail
        logger.info("XGBoost training complete.")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_lambdas(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (lambda_home, lambda_away) arrays."""
        if self.model_home is None or self.model_away is None:
            raise RuntimeError("Model not fitted.")
        lh = np.clip(self.model_home.predict(X), 0.05, 15.0)
        la = np.clip(self.model_away.predict(X), 0.05, 15.0)
        return lh, la

    def predict_result_proba(self, X: np.ndarray) -> np.ndarray:
        """Return shape (n, 3) array: P(home_win), P(draw), P(away_win)."""
        if self.model_result is None:
            raise RuntimeError("Model not fitted.")
        return self.model_result.predict_proba(X)

    def feature_importances(self) -> pd.DataFrame:
        if self.model_home is None:
            raise RuntimeError("Model not fitted.")
        return pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance_home": self.model_home.feature_importances_,
                "importance_away": self.model_away.feature_importances_,  # type: ignore[union-attr]
            }
        ).sort_values("importance_home", ascending=False)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path | None = None) -> Path:
        """
        Save using XGBoost's native UBJ format (save_model), not pickle/joblib.

        Pickling the sklearn wrapper embeds a raw memory buffer that is not
        guaranteed to round-trip across platforms (e.g. Linux-trained models
        failing to load on Windows with "input stream corrupted"). The native
        save_model()/load_model() format is documented by XGBoost as the
        portable, cross-platform-safe option.
        """
        out = directory or XGBOOST_MODEL_DIR
        out.mkdir(parents=True, exist_ok=True)
        self.model_home.save_model(str(out / _MODEL_HOME_FILE))  # type: ignore[union-attr]
        self.model_away.save_model(str(out / _MODEL_AWAY_FILE))  # type: ignore[union-attr]
        self.model_result.save_model(str(out / _MODEL_RESULT_FILE))  # type: ignore[union-attr]
        joblib.dump(self.feature_cols, out / "feature_cols.joblib")
        logger.info("Saved XGBoost models to %s", out)
        return out

    @classmethod
    def load(cls, directory: Path | None = None) -> "XGBoostPredictor":
        d = directory or XGBOOST_MODEL_DIR
        obj = cls()
        obj.model_home = xgb.XGBRegressor()
        obj.model_home.load_model(str(d / _MODEL_HOME_FILE))
        obj.model_away = xgb.XGBRegressor()
        obj.model_away.load_model(str(d / _MODEL_AWAY_FILE))
        obj.model_result = xgb.XGBClassifier()
        obj.model_result.load_model(str(d / _MODEL_RESULT_FILE))
        obj.feature_cols = joblib.load(d / "feature_cols.joblib")
        logger.info("Loaded XGBoost models from %s", d)
        return obj
