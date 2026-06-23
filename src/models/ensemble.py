"""
Ensemble combiner for Bayesian and XGBoost lambda predictions.

The ensemble computes a convex combination of the two models' expected-goal
estimates.  Both models operate independently — XGBoost does NOT use
Bayesian outputs as features, so the combination is a genuine diversity-based
ensemble rather than a two-stage pipeline.

Default weights (configurable):
    λ_final = 0.6 · λ_bayes + 0.4 · λ_xgb

These defaults were chosen to give more weight to the Bayesian model, which
has stronger inductive bias for low-data teams, while letting XGBoost
capture complex feature interactions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.config import ENSEMBLE_WEIGHT_BAYESIAN, ENSEMBLE_WEIGHT_XGBOOST

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Container for all ensemble outputs."""

    lambda_home: float
    lambda_away: float
    lambda_home_bayes: float
    lambda_away_bayes: float
    lambda_home_xgb: float
    lambda_away_xgb: float
    weight_bayes: float
    weight_xgb: float


class EnsembleModel:
    """
    Weighted-average ensemble of Bayesian and XGBoost λ estimates.

    Parameters
    ----------
    weight_bayes : float
        Weight for the Bayesian model (must sum to 1 with weight_xgb).
    weight_xgb : float
        Weight for the XGBoost model.
    """

    def __init__(
        self,
        weight_bayes: float = ENSEMBLE_WEIGHT_BAYESIAN,
        weight_xgb: float = ENSEMBLE_WEIGHT_XGBOOST,
    ) -> None:
        if abs(weight_bayes + weight_xgb - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_bayes + weight_xgb:.4f}"
            )
        self.weight_bayes = weight_bayes
        self.weight_xgb = weight_xgb

    def combine(
        self,
        lambda_home_bayes: float,
        lambda_away_bayes: float,
        lambda_home_xgb: float,
        lambda_away_xgb: float,
    ) -> EnsemblePrediction:
        """Return the weighted-average expected goals."""
        lh = self.weight_bayes * lambda_home_bayes + self.weight_xgb * lambda_home_xgb
        la = self.weight_bayes * lambda_away_bayes + self.weight_xgb * lambda_away_xgb

        # Guard against pathological inputs
        lh = float(np.clip(lh, 0.05, 15.0))
        la = float(np.clip(la, 0.05, 15.0))

        return EnsemblePrediction(
            lambda_home=lh,
            lambda_away=la,
            lambda_home_bayes=lambda_home_bayes,
            lambda_away_bayes=lambda_away_bayes,
            lambda_home_xgb=lambda_home_xgb,
            lambda_away_xgb=lambda_away_xgb,
            weight_bayes=self.weight_bayes,
            weight_xgb=self.weight_xgb,
        )

    def set_weights(self, weight_bayes: float, weight_xgb: float) -> None:
        """Update ensemble weights at runtime."""
        if abs(weight_bayes + weight_xgb - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self.weight_bayes = weight_bayes
        self.weight_xgb = weight_xgb
        logger.info(
            "Ensemble weights updated: bayes=%.2f  xgb=%.2f",
            weight_bayes,
            weight_xgb,
        )
