"""
Main predictor: orchestrates pipeline → ensemble → Poisson matrix.

Usage
-----
    predictor = MatchPredictor.load()
    result = predictor.predict("Argentina", "Brazil")
    print(result)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    ENSEMBLE_WEIGHT_BAYESIAN,
    ENSEMBLE_WEIGHT_XGBOOST,
)
from src.features.pipeline import FEATURE_COLUMNS, FeaturePipeline
from src.models.bayesian import BayesianHierarchicalModel
from src.models.ensemble import EnsembleModel
from src.models.xgboost_model import XGBoostPredictor
from src.prediction.events import MatchEventsEstimate, estimate_match_events
from src.prediction.patterns import PatternReport, analyze_patterns
from src.prediction.poisson import PoissonPrediction, predict_from_lambdas

logger = logging.getLogger(__name__)


@dataclass
class MatchPrediction:
    """Complete prediction output for a single match."""

    home_team: str
    away_team: str

    # Match result
    home_win: float
    draw: float
    away_win: float

    # Expected goals
    expected_goals_home: float
    expected_goals_away: float

    # Most likely exact score
    most_likely_score: str

    # Goals markets
    btts: float
    over_0_5: float
    over_1_5: float
    over_2_5: float
    over_3_5: float

    # Score distribution
    score_probabilities: dict[str, float]

    # Match events (shots, corners, cards, etc.)
    events: MatchEventsEstimate

    # Historical pattern analysis
    patterns: Optional[PatternReport]

    # Decomposition
    lambda_bayes_home: float
    lambda_bayes_away: float
    lambda_xgb_home: float
    lambda_xgb_away: float
    model_weights: dict[str, float]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class MatchPredictor:
    """
    High-level predictor that combines all system components.

    After training, call *predict(home_team, away_team)* for inference.
    """

    def __init__(
        self,
        pipeline: Optional[FeaturePipeline] = None,
        bayesian: Optional[BayesianHierarchicalModel] = None,
        xgboost: Optional[XGBoostPredictor] = None,
        weight_bayes: float = ENSEMBLE_WEIGHT_BAYESIAN,
        weight_xgb: float = ENSEMBLE_WEIGHT_XGBOOST,
    ) -> None:
        self.pipeline = pipeline
        self.bayesian = bayesian
        self.xgboost = xgboost
        self.ensemble = EnsembleModel(weight_bayes, weight_xgb)
        self._feature_df: Optional[pd.DataFrame] = None  # full trained dataset

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = False,
        tournament: str = "Friendly",
        date: Optional[pd.Timestamp] = None,
    ) -> MatchPrediction:
        """
        Generate a full prediction for an upcoming match.

        Parameters
        ----------
        home_team, away_team : str
            Official team names (must match dataset spelling).
        is_neutral : bool
            True if the match is played at a neutral venue.
        tournament : str
            Tournament name (affects tournament encoding feature).
        date : pd.Timestamp, optional
            Match date for temporal features (defaults to today).
        """
        if self.pipeline is None or self.xgboost is None:
            raise RuntimeError("Predictor not initialised.  Call MatchPredictor.load().")

        if date is None:
            date = pd.Timestamp.now()

        # ── Bayesian λ (uses posterior mean parameters) ───────────────────────
        if self.bayesian is not None:
            lh_bayes, la_bayes = self.bayesian.predict_lambdas(
                home_team, away_team, is_neutral
            )
        else:
            lh_bayes, la_bayes = 0.0, 0.0  # unused when weight_bayes=0
        logger.debug("Bayes lambdas: home=%.3f  away=%.3f", lh_bayes, la_bayes)

        # ── XGBoost λ (uses feature vector) ──────────────────────────────────
        feature_row = self._build_feature_row(home_team, away_team, is_neutral, tournament, date)
        X = feature_row[self.xgboost.feature_cols].fillna(0.0).values.reshape(1, -1)
        lh_xgb_arr, la_xgb_arr = self.xgboost.predict_lambdas(X)
        lh_xgb = float(lh_xgb_arr[0])
        la_xgb = float(la_xgb_arr[0])
        logger.debug("XGBoost lambdas: home=%.3f  away=%.3f", lh_xgb, la_xgb)

        # ── Ensemble ──────────────────────────────────────────────────────────
        ens = self.ensemble.combine(lh_bayes, la_bayes, lh_xgb, la_xgb)

        # ── Poisson matrix → all probabilities ────────────────────────────────
        poisson_pred: PoissonPrediction = predict_from_lambdas(ens.lambda_home, ens.lambda_away)

        # ── Match events (shots, corners, cards…) ─────────────────────────────
        tournament_importance = float(
            feature_row.get("tournament_importance", 1.0)
            if hasattr(feature_row, "get") else 1.0
        )
        events = estimate_match_events(
            lambda_home=ens.lambda_home,
            lambda_away=ens.lambda_away,
            tournament_importance=tournament_importance,
        )

        # ── Historical patterns ───────────────────────────────────────────────
        pat_report = None
        if self._feature_df is not None:
            elo_home_val = self.pipeline.elo_system.get_rating(home_team)  # type: ignore
            elo_away_val = self.pipeline.elo_system.get_rating(away_team)  # type: ignore
            elo_diff_val = elo_home_val - elo_away_val
            tournament_cat = str(feature_row.get("tournament_category", "friendly")
                                 if hasattr(feature_row, "get") else "friendly")
            try:
                pat_report = analyze_patterns(
                    df=self._feature_df,
                    elo_diff=elo_diff_val,
                    lambda_home=ens.lambda_home,
                    lambda_away=ens.lambda_away,
                    is_neutral=is_neutral,
                    tournament_category=tournament_cat,
                    home_team=home_team,
                    away_team=away_team,
                )
            except Exception as exc:
                logger.warning("Pattern analysis failed: %s", exc)

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win=poisson_pred.home_win,
            draw=poisson_pred.draw,
            away_win=poisson_pred.away_win,
            expected_goals_home=round(ens.lambda_home, 3),
            expected_goals_away=round(ens.lambda_away, 3),
            most_likely_score=poisson_pred.most_likely_score,
            btts=poisson_pred.btts,
            over_0_5=poisson_pred.over_0_5,
            over_1_5=poisson_pred.over_1_5,
            over_2_5=poisson_pred.over_2_5,
            over_3_5=poisson_pred.over_3_5,
            score_probabilities=poisson_pred.score_probs,
            events=events,
            patterns=pat_report,
            lambda_bayes_home=round(lh_bayes, 3),
            lambda_bayes_away=round(la_bayes, 3),
            lambda_xgb_home=round(lh_xgb, 3),
            lambda_xgb_away=round(la_xgb, 3),
            model_weights={"bayesian": ens.weight_bayes, "xgboost": ens.weight_xgb},
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_feature_row(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool,
        tournament: str,
        date: pd.Timestamp,
    ) -> pd.Series:
        """
        Build a single feature row for inference using the latest known
        statistics for each team from the training dataset.
        """
        if self._feature_df is None:
            raise RuntimeError("Feature DataFrame not set.  Ensure pipeline is loaded.")

        df = self._feature_df

        home_feat = self._latest_team_features(df, home_team, "home", date)
        away_feat = self._latest_team_features(df, away_team, "away", date)

        elo_home = self.pipeline.elo_system.get_rating(home_team)  # type: ignore[union-attr]
        elo_away = self.pipeline.elo_system.get_rating(away_team)  # type: ignore[union-attr]

        # Head-to-head from historical data
        h2h = self._compute_h2h(df, home_team, away_team, date)

        row: dict = {
            **home_feat,
            **away_feat,
            **h2h,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home - elo_away,
            "neutral": int(is_neutral),
            "tournament": tournament,
            "date": date,
        }

        temp = pd.DataFrame([row])
        from src.features.encoders import add_temporal_features, encode_tournament
        temp = encode_tournament(temp)
        temp = add_temporal_features(temp)

        return temp.iloc[0]

    @staticmethod
    def _compute_h2h(
        df: pd.DataFrame,
        home: str,
        away: str,
        before: pd.Timestamp,
    ) -> dict:
        """Compute head-to-head stats for the upcoming match."""
        from src.config import H2H_WINDOW
        mask = (
            (
                ((df["home_team"] == home) & (df["away_team"] == away))
                | ((df["home_team"] == away) & (df["away_team"] == home))
            )
            & (df["date"] < before)
        )
        h2h = df[mask].tail(H2H_WINDOW)
        if len(h2h) == 0:
            return {
                "h2h_played": 0.0, "h2h_home_wins": 0.0,
                "h2h_draws": 0.0, "h2h_away_wins": 0.0, "h2h_avg_goals": 0.0,
            }
        home_wins = (
            ((h2h["home_team"] == home) & (h2h["home_score"] > h2h["away_score"]))
            | ((h2h["away_team"] == home) & (h2h["away_score"] > h2h["home_score"]))
        ).sum()
        away_wins = (
            ((h2h["home_team"] == away) & (h2h["home_score"] > h2h["away_score"]))
            | ((h2h["away_team"] == away) & (h2h["away_score"] > h2h["home_score"]))
        ).sum()
        return {
            "h2h_played": float(len(h2h)),
            "h2h_home_wins": float(home_wins),
            "h2h_draws": float(len(h2h) - home_wins - away_wins),
            "h2h_away_wins": float(away_wins),
            "h2h_avg_goals": float((h2h["home_score"] + h2h["away_score"]).mean()),
        }

    def _latest_team_features(
        self,
        df: pd.DataFrame,
        team: str,
        role: str,
        before: pd.Timestamp,
    ) -> dict:
        """
        Extract form/strength features for *team* from their most recent row
        in the feature DataFrame where they appeared as *role* (home/away).
        """
        col = f"{role}_team"
        opposing_role = "away" if role == "home" else "home"
        mask = (df[col] == team) & (df["date"] < before)
        rows = df[mask]

        if rows.empty:
            # Team not seen as home/away — try the other side
            col2 = f"{opposing_role}_team"
            mask2 = (df[col2] == team) & (df["date"] < before)
            rows = df[mask2]

        if rows.empty:
            # Completely unknown team — return zero features
            return {f"{role}_{stat}": 0.0 for stat in [
                "wins_last5", "draws_last5", "losses_last5",
                "gf_last5", "ga_last5", "gd_last5",
                "win_rate_last5", "avg_gf_last5", "avg_ga_last5",
                "wins_last10", "draws_last10", "losses_last10",
                "gf_last10", "ga_last10", "gd_last10",
                "win_rate_last10", "avg_gf_last10", "avg_ga_last10",
                "attack_strength", "defense_strength", "days_rest",
            ]}

        last = rows.sort_values("date").iloc[-1]

        # Extract features — handle both home and away perspective
        feature_names = [
            "wins_last5", "draws_last5", "losses_last5",
            "gf_last5", "ga_last5", "gd_last5",
            "win_rate_last5", "avg_gf_last5", "avg_ga_last5",
            "wins_last10", "draws_last10", "losses_last10",
            "gf_last10", "ga_last10", "gd_last10",
            "win_rate_last10", "avg_gf_last10", "avg_ga_last10",
            "attack_strength", "defense_strength", "days_rest",
        ]
        # The DataFrame stores features with the prefix matching the team's role
        # in THAT row.  We need to re-label them for the current prediction role.
        src_prefix = "home" if col == "home_team" else "away"
        result: dict = {}
        for stat in feature_names:
            src_col = f"{src_prefix}_{stat}"
            result[f"{role}_{stat}"] = float(last.get(src_col, 0.0))

        # Head-to-head is match-specific so we zero it; predictor will compute it
        return result

    # ── Load / Save ───────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        weight_bayes: float = ENSEMBLE_WEIGHT_BAYESIAN,
        weight_xgb: float = ENSEMBLE_WEIGHT_XGBOOST,
    ) -> "MatchPredictor":
        """
        Load all trained components from their default directories.

        If the Bayesian model is not found (e.g. --skip-bayesian was used),
        falls back to XGBoost-only mode (weight_xgb = 1.0).
        """
        from src.features.pipeline import FeaturePipeline, load_features

        pipeline = FeaturePipeline.load()
        xgboost = XGBoostPredictor.load()

        bayesian = None
        try:
            bayesian = BayesianHierarchicalModel.load()
        except FileNotFoundError:
            logger.warning(
                "Bayesian model not found — running in XGBoost-only mode. "
                "Run 'python train.py' (without --skip-bayesian) to train it."
            )
            weight_bayes = 0.0
            weight_xgb = 1.0

        obj = cls(
            pipeline=pipeline,
            bayesian=bayesian,
            xgboost=xgboost,
            weight_bayes=weight_bayes,
            weight_xgb=weight_xgb,
        )
        obj._feature_df = load_features()
        logger.info("MatchPredictor loaded (bayes=%.0f%% xgb=%.0f%%).", weight_bayes * 100, weight_xgb * 100)
        return obj
