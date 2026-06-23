"""Unit tests for model components (XGBoost, Ensemble)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import EnsembleModel
from src.models.xgboost_model import XGBoostPredictor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_feature_df() -> pd.DataFrame:
    """Minimal feature DataFrame for XGBoost smoke tests."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "elo_home": rng.normal(1500, 100, n),
        "elo_away": rng.normal(1500, 100, n),
        "elo_diff": rng.normal(0, 150, n),
        "home_wins_last5": rng.integers(0, 5, n).astype(float),
        "home_draws_last5": rng.integers(0, 3, n).astype(float),
        "home_losses_last5": rng.integers(0, 3, n).astype(float),
        "home_gf_last5": rng.integers(0, 10, n).astype(float),
        "home_ga_last5": rng.integers(0, 10, n).astype(float),
        "home_gd_last5": rng.normal(0, 3, n),
        "home_win_rate_last5": rng.random(n),
        "home_avg_gf_last5": rng.random(n) * 3,
        "home_avg_ga_last5": rng.random(n) * 3,
        "away_wins_last5": rng.integers(0, 5, n).astype(float),
        "away_draws_last5": rng.integers(0, 3, n).astype(float),
        "away_losses_last5": rng.integers(0, 3, n).astype(float),
        "away_gf_last5": rng.integers(0, 10, n).astype(float),
        "away_ga_last5": rng.integers(0, 10, n).astype(float),
        "away_gd_last5": rng.normal(0, 3, n),
        "away_win_rate_last5": rng.random(n),
        "away_avg_gf_last5": rng.random(n) * 3,
        "away_avg_ga_last5": rng.random(n) * 3,
        "home_wins_last10": rng.integers(0, 8, n).astype(float),
        "home_draws_last10": rng.integers(0, 4, n).astype(float),
        "home_losses_last10": rng.integers(0, 4, n).astype(float),
        "home_gf_last10": rng.integers(0, 20, n).astype(float),
        "home_ga_last10": rng.integers(0, 20, n).astype(float),
        "home_gd_last10": rng.normal(0, 5, n),
        "home_win_rate_last10": rng.random(n),
        "home_avg_gf_last10": rng.random(n) * 3,
        "home_avg_ga_last10": rng.random(n) * 3,
        "away_wins_last10": rng.integers(0, 8, n).astype(float),
        "away_draws_last10": rng.integers(0, 4, n).astype(float),
        "away_losses_last10": rng.integers(0, 4, n).astype(float),
        "away_gf_last10": rng.integers(0, 20, n).astype(float),
        "away_ga_last10": rng.integers(0, 20, n).astype(float),
        "away_gd_last10": rng.normal(0, 5, n),
        "away_win_rate_last10": rng.random(n),
        "away_avg_gf_last10": rng.random(n) * 3,
        "away_avg_ga_last10": rng.random(n) * 3,
        "home_attack_strength": rng.random(n) * 2 + 0.5,
        "home_defense_strength": rng.random(n) * 2 + 0.5,
        "away_attack_strength": rng.random(n) * 2 + 0.5,
        "away_defense_strength": rng.random(n) * 2 + 0.5,
        "home_days_rest": rng.integers(3, 200, n).astype(float),
        "away_days_rest": rng.integers(3, 200, n).astype(float),
        "h2h_played": rng.integers(0, 10, n).astype(float),
        "h2h_home_wins": rng.integers(0, 5, n).astype(float),
        "h2h_draws": rng.integers(0, 3, n).astype(float),
        "h2h_away_wins": rng.integers(0, 5, n).astype(float),
        "h2h_avg_goals": rng.random(n) * 3 + 1,
        "tournament_code": rng.integers(0, 6, n),
        "tournament_importance": rng.choice([1.0, 2.0, 3.5, 4.0], n),
        "neutral": rng.integers(0, 2, n),
        "year": rng.integers(2000, 2024, n),
        "month": rng.integers(1, 13, n),
        # Targets
        "home_score": rng.poisson(1.4, n),
        "away_score": rng.poisson(1.1, n),
        "sample_weight": np.ones(n),
    })
    df["result"] = np.sign(df["home_score"] - df["away_score"]).astype(int)
    return df


# ── Ensemble tests ────────────────────────────────────────────────────────────

class TestEnsembleModel:
    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            EnsembleModel(weight_bayes=0.7, weight_xgb=0.5)

    def test_combine_basic(self) -> None:
        ens = EnsembleModel(0.6, 0.4)
        result = ens.combine(2.0, 1.0, 1.0, 2.0)
        assert result.lambda_home == pytest.approx(0.6 * 2.0 + 0.4 * 1.0)
        assert result.lambda_away == pytest.approx(0.6 * 1.0 + 0.4 * 2.0)

    def test_combine_clamps_to_bounds(self) -> None:
        ens = EnsembleModel(0.5, 0.5)
        result = ens.combine(100.0, 0.0, 100.0, 0.0)
        assert result.lambda_home <= 15.0
        assert result.lambda_away >= 0.05

    def test_set_weights(self) -> None:
        ens = EnsembleModel(0.6, 0.4)
        ens.set_weights(0.3, 0.7)
        assert ens.weight_bayes == 0.3
        assert ens.weight_xgb == 0.7

    def test_full_weight_on_one_model(self) -> None:
        ens = EnsembleModel(1.0, 0.0)
        result = ens.combine(2.5, 1.3, 9.9, 9.9)
        assert result.lambda_home == pytest.approx(2.5)
        assert result.lambda_away == pytest.approx(1.3)


# ── XGBoost tests ─────────────────────────────────────────────────────────────

class TestXGBoostPredictor:
    def test_fit_and_predict(self, tiny_feature_df: pd.DataFrame) -> None:
        model = XGBoostPredictor()
        model.fit(tiny_feature_df, optimize=False)

        avail = [c for c in model.feature_cols if c in tiny_feature_df.columns]
        X = tiny_feature_df[avail].fillna(0.0).values
        lh, la = model.predict_lambdas(X)

        assert lh.shape == (len(tiny_feature_df),)
        assert la.shape == (len(tiny_feature_df),)
        assert (lh > 0).all(), "All lambda_home predictions must be positive"
        assert (la > 0).all(), "All lambda_away predictions must be positive"

    def test_lambdas_in_reasonable_range(self, tiny_feature_df: pd.DataFrame) -> None:
        model = XGBoostPredictor()
        model.fit(tiny_feature_df, optimize=False)
        avail = [c for c in model.feature_cols if c in tiny_feature_df.columns]
        X = tiny_feature_df[avail].fillna(0.0).values
        lh, la = model.predict_lambdas(X)
        assert lh.max() <= 15.0
        assert lh.min() >= 0.05

    def test_result_classifier_shape(self, tiny_feature_df: pd.DataFrame) -> None:
        model = XGBoostPredictor()
        model.fit(tiny_feature_df, optimize=False)
        avail = [c for c in model.feature_cols if c in tiny_feature_df.columns]
        X = tiny_feature_df[avail].fillna(0.0).values
        proba = model.predict_result_proba(X)
        assert proba.shape == (len(tiny_feature_df), 3)
        # Probabilities sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importances(self, tiny_feature_df: pd.DataFrame) -> None:
        model = XGBoostPredictor()
        model.fit(tiny_feature_df, optimize=False)
        imp = model.feature_importances()
        assert "feature" in imp.columns
        assert "importance_home" in imp.columns
        assert len(imp) > 0

    def test_raises_before_fit(self) -> None:
        model = XGBoostPredictor()
        with pytest.raises(RuntimeError):
            model.predict_lambdas(np.zeros((1, 10)))
