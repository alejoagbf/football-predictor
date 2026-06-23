"""
Feature engineering pipeline.

Orchestrates ELO computation, form features, and categorical encoding
to produce the full feature matrix used by both models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, FEATURES_FILE, XGBOOST_TEMPORAL_DECAY
from src.features.elo import EloSystem
from src.features.encoders import add_temporal_features, encode_tournament
from src.features.form import compute_form_features

logger = logging.getLogger(__name__)

# All feature columns produced by the pipeline (used by both models)
FEATURE_COLUMNS: list[str] = [
    # ELO
    "elo_home",
    "elo_away",
    "elo_diff",
    # Form – last 5
    "home_wins_last5",
    "home_draws_last5",
    "home_losses_last5",
    "home_gf_last5",
    "home_ga_last5",
    "home_gd_last5",
    "home_win_rate_last5",
    "home_avg_gf_last5",
    "home_avg_ga_last5",
    "away_wins_last5",
    "away_draws_last5",
    "away_losses_last5",
    "away_gf_last5",
    "away_ga_last5",
    "away_gd_last5",
    "away_win_rate_last5",
    "away_avg_gf_last5",
    "away_avg_ga_last5",
    # Form – last 10
    "home_wins_last10",
    "home_draws_last10",
    "home_losses_last10",
    "home_gf_last10",
    "home_ga_last10",
    "home_gd_last10",
    "home_win_rate_last10",
    "home_avg_gf_last10",
    "home_avg_ga_last10",
    "away_wins_last10",
    "away_draws_last10",
    "away_losses_last10",
    "away_gf_last10",
    "away_ga_last10",
    "away_gd_last10",
    "away_win_rate_last10",
    "away_avg_gf_last10",
    "away_avg_ga_last10",
    # Strength
    "home_attack_strength",
    "home_defense_strength",
    "away_attack_strength",
    "away_defense_strength",
    # Rest
    "home_days_rest",
    "away_days_rest",
    # Head-to-head
    "h2h_played",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "h2h_avg_goals",
    # Tournament / venue
    "tournament_code",
    "tournament_importance",
    "neutral",
    # Temporal
    "year",
    "month",
]

# Rename map from form.py output → canonical feature column names
_FORM_RENAME: dict[str, str] = {}
for _prefix in ("home", "away"):
    for _w in (5, 10):
        for _stat in ("matches", "wins", "draws", "losses", "gf", "ga", "gd", "win_rate", "avg_gf", "avg_ga"):
            _FORM_RENAME[f"{_prefix}_{_stat}_last{_w}"] = f"{_prefix}_{_stat}_last{_w}"


class FeaturePipeline:
    """
    End-to-end feature pipeline.

    Usage::

        pipeline = FeaturePipeline()
        feature_df = pipeline.fit_transform(raw_df)   # training
        feature_df = pipeline.transform(raw_df)        # inference (reuses ELO state)
    """

    def __init__(self) -> None:
        self.elo_system = EloSystem()
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features from scratch on the full dataset.

        Also persists the ELO state so that new predictions can call
        *transform* with up-to-date ratings.
        """
        logger.info("Running full feature pipeline on %d rows …", len(df))
        df = df.sort_values("date").reset_index(drop=True).copy()

        # Step 1: ELO (builds ratings chronologically from scratch)
        self.elo_system = EloSystem()
        df = self.elo_system.compute_elo_history(df)

        # Step 2: Rolling form (O(n * teams) — takes a few minutes on 49k rows)
        df = compute_form_features(df)

        # Step 3: Tournament & temporal encodings
        df = encode_tournament(df)
        df = add_temporal_features(df)

        # Step 4: neutral as int
        df["neutral"] = df["neutral"].astype(int)

        # Step 5: compute sample weights for XGBoost (exponential temporal decay)
        today_days = df["days_from_epoch"].max()
        df["sample_weight"] = np.exp(
            -XGBOOST_TEMPORAL_DECAY * (today_days - df["days_from_epoch"])
        )

        self._fitted = True
        logger.info("Pipeline complete. Shape: %s", df.shape)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features to *df* using the already-fitted ELO state.

        Intended for computing features for a *single* upcoming match
        where ratings have already been built by fit_transform.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        df = df.copy()
        df = encode_tournament(df)
        df = add_temporal_features(df)
        df["neutral"] = df["neutral"].astype(int)
        return df

    def get_feature_matrix(
        self, df: pd.DataFrame, feature_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Return X matrix with only feature columns, NaN → 0."""
        cols = feature_cols or FEATURE_COLUMNS
        available = [c for c in cols if c in df.columns]
        X = df[available].fillna(0.0)
        return X

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = path or DATA_PROCESSED_DIR / "pipeline.joblib"
        joblib.dump(self, out)
        logger.info("Saved pipeline to %s", out)
        return out

    @classmethod
    def load(cls, path: Path | None = None) -> "FeaturePipeline":
        p = path or DATA_PROCESSED_DIR / "pipeline.joblib"
        obj: FeaturePipeline = joblib.load(p)
        logger.info("Loaded pipeline from %s", p)
        return obj


def save_features(df: pd.DataFrame, path: Path | None = None) -> None:
    out = path or FEATURES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Saved feature DataFrame to %s", out)


def load_features(path: Path | None = None) -> pd.DataFrame:
    p = path or FEATURES_FILE
    df = pd.read_parquet(p)
    logger.info("Loaded %d rows from %s", len(df), p)
    return df
