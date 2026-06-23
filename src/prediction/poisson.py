"""
Poisson score-matrix and derived market probabilities.

Given λ_home and λ_away (expected goals per team), this module builds a
complete joint probability matrix under the independence assumption:

    P(home_goals = i, away_goals = j) = Poisson(i; λ_home) · Poisson(j; λ_away)

The matrix covers scores 0–POISSON_MAX_GOALS for each side (default 0–7).
All market probabilities (1X2, over/under, BTTS) are derived by marginalising
over this matrix.

Limitation / known bias
-----------------------
The classic Bivariate Poisson extension (Dixon & Coles 1997) adds a
correlation correction that slightly boosts 0-0 and reduces 1-0 / 0-1
probabilities for low-scoring games.  This module implements the simpler
independent-Poisson version; the DC correction can be added as an extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import poisson

from src.config import POISSON_MAX_GOALS


@dataclass
class PoissonPrediction:
    """All probability outputs derived from the score matrix."""

    # Expected goals
    lambda_home: float
    lambda_away: float

    # Match result probabilities
    home_win: float
    draw: float
    away_win: float

    # Over/under markets
    over_0_5: float
    over_1_5: float
    over_2_5: float
    over_3_5: float

    # Both teams to score
    btts: float

    # Most likely exact score
    most_likely_score: str

    # Full score distribution {score_str: probability}
    score_probs: dict[str, float]


def build_score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = POISSON_MAX_GOALS,
) -> np.ndarray:
    """
    Compute the (max_goals+1) × (max_goals+1) joint probability matrix.

    Row index = home goals, column index = away goals.
    Uses scipy.stats.poisson for numerically stable PMF values.
    """
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    matrix = np.outer(home_probs, away_probs)
    # Renormalise to account for truncation at max_goals
    matrix /= matrix.sum()
    return matrix


def matrix_to_probabilities(
    matrix: np.ndarray,
    lambda_home: float,
    lambda_away: float,
) -> PoissonPrediction:
    """
    Derive all market probabilities from the score matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (max_goals+1, max_goals+1) joint probability matrix.
    lambda_home, lambda_away : float
        Expected goals (passed through to the output dataclass).
    """
    n = matrix.shape[0]  # = max_goals + 1

    # 1X2
    home_win = float(np.tril(matrix, k=-1).sum())   # home > away  (below diagonal)
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())    # away > home  (above diagonal)

    # Total goals matrix (vectorised)
    idx = np.arange(n)
    total_goals = idx[:, None] + idx[None, :]  # shape (n, n)

    over_0_5 = float(np.sum(matrix[total_goals >= 1]))
    over_1_5 = float(np.sum(matrix[total_goals >= 2]))
    over_2_5 = float(np.sum(matrix[total_goals >= 3]))
    over_3_5 = float(np.sum(matrix[total_goals >= 4]))

    # BTTS: both teams score at least 1 goal
    # = 1 - P(home=0) - P(away=0) + P(home=0 AND away=0)
    btts = float(1.0 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0])

    # Most likely score
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_likely_score = f"{max_idx[0]}-{max_idx[1]}"

    # Top-N score distribution (filter to scores with >0.1% probability)
    score_probs: dict[str, float] = {}
    for i in range(n):
        for j in range(n):
            p = float(matrix[i, j])
            if p > 0.0005:  # incluir todos los marcadores con >= 0.05%
                score_probs[f"{i}-{j}"] = round(p, 4)

    return PoissonPrediction(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        home_win=round(home_win, 4),
        draw=round(draw, 4),
        away_win=round(away_win, 4),
        over_0_5=round(over_0_5, 4),
        over_1_5=round(over_1_5, 4),
        over_2_5=round(over_2_5, 4),
        over_3_5=round(over_3_5, 4),
        btts=round(btts, 4),
        most_likely_score=most_likely_score,
        score_probs=score_probs,
    )


def predict_from_lambdas(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = POISSON_MAX_GOALS,
) -> PoissonPrediction:
    """Convenience function: build matrix then compute all probabilities."""
    matrix = build_score_matrix(lambda_home, lambda_away, max_goals)
    return matrix_to_probabilities(matrix, lambda_home, lambda_away)
