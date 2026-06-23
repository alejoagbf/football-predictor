"""
Walk-forward temporal validation.

Classic k-fold with random splitting would leak future information because
the feature engineering (ELO, form) is inherently time-dependent.

This module implements expanding-window walk-forward validation:

    Fold 1:  train 1990–2015,  validate 2016
    Fold 2:  train 1990–2016,  validate 2017
    …
    Fold N:  train 1990–2023,  validate 2024

For each fold a *fresh* ELO history and XGBoost model is trained from
scratch on the expanding window, so the ELO ratings are always computed
without any future data contaminating the features.

Note: the Bayesian model is excluded from walk-forward validation because
full NUTS re-training per fold would be prohibitively slow.  A surrogate
XGBoost-only validation is used instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)

from src.features.elo import EloSystem
from src.features.encoders import add_temporal_features, encode_tournament
from src.features.form import compute_form_features
from src.models.xgboost_model import XGBoostPredictor
from src.prediction.poisson import predict_from_lambdas

logger = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    """Evaluation metrics for a single validation fold."""

    year: int
    n_train: int
    n_val: int

    # Classification
    accuracy: float
    log_loss_result: float
    brier_home: float
    brier_draw: float
    brier_away: float

    # Regression (goals)
    rmse_home_goals: float
    mae_home_goals: float
    rmse_away_goals: float
    mae_away_goals: float

    # Calibration (returned as strings for easy serialisation)
    calibration_note: str = ""


@dataclass
class ValidationReport:
    """Aggregated results across all folds."""

    folds: list[FoldMetrics] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "year": f.year,
                "n_val": f.n_val,
                "accuracy": f.accuracy,
                "log_loss": f.log_loss_result,
                "brier_home": f.brier_home,
                "rmse_home": f.rmse_home_goals,
                "mae_home": f.mae_home_goals,
                "rmse_away": f.rmse_away_goals,
                "mae_away": f.mae_away_goals,
            }
            for f in self.folds
        ])

    def mean_metrics(self) -> dict[str, float]:
        df = self.summary()
        return df.drop(columns=["year", "n_val"]).mean().to_dict()


def _build_features_for_fold(df_train: pd.DataFrame) -> pd.DataFrame:
    """Re-build all features from scratch for one fold's training data."""
    elo = EloSystem()
    df = elo.compute_elo_history(df_train.copy())
    df = compute_form_features(df)
    df = encode_tournament(df)
    df = add_temporal_features(df)
    df["neutral"] = df["neutral"].astype(int)
    # Temporal weights (decay from fold's own max date)
    from src.config import XGBOOST_TEMPORAL_DECAY
    days_max = df["days_from_epoch"].max()
    df["sample_weight"] = np.exp(
        -XGBOOST_TEMPORAL_DECAY * (days_max - df["days_from_epoch"])
    )
    return df


def _poisson_probabilities(
    df: pd.DataFrame,
    model: XGBoostPredictor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (proba_matrix, y_true_result) for *df* given fitted *model*.

    proba_matrix : shape (n, 3) — P(home_win), P(draw), P(away_win)
    y_true_result : 0=home_win, 1=draw, 2=away_win
    """
    from src.features.pipeline import FEATURE_COLUMNS
    avail = [c for c in model.feature_cols if c in df.columns]
    X = df[avail].fillna(0.0).values

    lh, la = model.predict_lambdas(X)

    probas = np.zeros((len(df), 3))
    for i in range(len(df)):
        pred = predict_from_lambdas(float(lh[i]), float(la[i]))
        probas[i] = [pred.home_win, pred.draw, pred.away_win]

    y_true = df["result"].map({1: 0, 0: 1, -1: 2}).fillna(1).values.astype(int)
    return probas, y_true


def run_walk_forward_validation(
    df_full: pd.DataFrame,
    start_year: int = 2016,
    train_start_year: int = 1990,
    optimize_xgb: bool = False,   # set True for production (slower)
    n_optuna_trials: int = 30,
) -> ValidationReport:
    """
    Execute expanding-window walk-forward validation.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full dataset as returned by load_results(), *before* any features.
    start_year : int
        First year used as validation window.
    train_start_year : int
        Earliest year included in training (avoids very old noisy data).
    optimize_xgb : bool
        Whether to run Optuna per fold (slow) or use default hyperparams.
    n_optuna_trials : int
        Optuna trials per fold when optimize_xgb=True.
    """
    report = ValidationReport()
    max_year = df_full["date"].dt.year.max()

    for val_year in range(start_year, max_year + 1):
        logger.info("=== Fold: validate %d ===", val_year)

        train_mask = (df_full["date"].dt.year < val_year) & (
            df_full["date"].dt.year >= train_start_year
        )
        val_mask = df_full["date"].dt.year == val_year

        df_train_raw = df_full[train_mask].copy()
        df_val_raw = df_full[val_mask].copy()

        if len(df_val_raw) == 0:
            logger.warning("No data for year %d — skipping.", val_year)
            continue

        logger.info("  Training rows: %d  |  Validation rows: %d", len(df_train_raw), len(df_val_raw))

        # Build features for training fold
        df_train = _build_features_for_fold(df_train_raw)

        # For validation we need features computed ON the training ELO state,
        # then forward-propagated through the val set.
        # We re-run ELO on train+val together but only use val rows for scoring.
        df_all_raw = df_full[
            (df_full["date"].dt.year >= train_start_year)
            & (df_full["date"].dt.year <= val_year)
        ].copy()
        elo_full = EloSystem()
        df_all = elo_full.compute_elo_history(df_all_raw)
        df_all = compute_form_features(df_all)
        df_all = encode_tournament(df_all)
        df_all = add_temporal_features(df_all)
        df_all["neutral"] = df_all["neutral"].astype(int)

        from src.config import XGBOOST_TEMPORAL_DECAY
        days_max = df_all["days_from_epoch"].max()
        df_all["sample_weight"] = np.exp(
            -XGBOOST_TEMPORAL_DECAY * (days_max - df_all["days_from_epoch"])
        )

        df_val = df_all[df_all["date"].dt.year == val_year]

        # Train XGBoost on training fold features
        model = XGBoostPredictor()
        model.fit(df_train, optimize=optimize_xgb, n_trials=n_optuna_trials)

        # Get Poisson probabilities for val set
        probas, y_true = _poisson_probabilities(df_val, model)
        y_pred_cls = np.argmax(probas, axis=1)

        # Goal predictions
        avail = [c for c in model.feature_cols if c in df_val.columns]
        X_val = df_val[avail].fillna(0.0).values
        lh_pred, la_pred = model.predict_lambdas(X_val)

        y_home_true = df_val["home_score"].values
        y_away_true = df_val["away_score"].values

        metrics = FoldMetrics(
            year=val_year,
            n_train=len(df_train),
            n_val=len(df_val),
            accuracy=float(accuracy_score(y_true, y_pred_cls)),
            log_loss_result=float(log_loss(y_true, probas, labels=[0, 1, 2])),
            brier_home=float(brier_score_loss(y_true == 0, probas[:, 0])),
            brier_draw=float(brier_score_loss(y_true == 1, probas[:, 1])),
            brier_away=float(brier_score_loss(y_true == 2, probas[:, 2])),
            rmse_home_goals=float(np.sqrt(mean_squared_error(y_home_true, lh_pred))),
            mae_home_goals=float(mean_absolute_error(y_home_true, lh_pred)),
            rmse_away_goals=float(np.sqrt(mean_squared_error(y_away_true, la_pred))),
            mae_away_goals=float(mean_absolute_error(y_away_true, la_pred)),
        )
        report.folds.append(metrics)
        logger.info(
            "  Accuracy: %.3f  |  LogLoss: %.3f  |  RMSE_h: %.3f  |  RMSE_a: %.3f",
            metrics.accuracy,
            metrics.log_loss_result,
            metrics.rmse_home_goals,
            metrics.rmse_away_goals,
        )

    logger.info("Walk-forward validation complete.  Mean metrics:")
    for k, v in report.mean_metrics().items():
        logger.info("  %s: %.4f", k, v)

    return report
