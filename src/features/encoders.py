"""
Tournament and categorical feature encoders.

Maps raw tournament names to a fixed set of categories and provides
ordinal encoding for model consumption.
"""

from __future__ import annotations

import pandas as pd

from src.config import TOURNAMENT_CATEGORIES, TOURNAMENT_CATEGORY_DEFAULT

# Ordered list of categories used for encoding (index = numeric code)
CATEGORY_ORDER: list[str] = [
    "friendly",
    "qualification",
    "nations_league",
    "continental",
    "world_cup",
    "olympics",
    "other",
]

# Importance weight associated with each category (used as an extra feature)
CATEGORY_IMPORTANCE: dict[str, float] = {
    "friendly": 1.0,
    "qualification": 2.0,
    "nations_league": 2.5,
    "continental": 3.5,
    "world_cup": 4.0,
    "olympics": 2.5,
    "other": 1.5,
}


def classify_tournament(tournament: str) -> str:
    """Map a raw tournament name to its canonical category string."""
    t_lower = tournament.lower()
    for pattern, category in TOURNAMENT_CATEGORIES.items():
        if pattern.lower() in t_lower:
            return category
    return TOURNAMENT_CATEGORY_DEFAULT


def encode_tournament(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add *tournament_category*, *tournament_code*, and *tournament_importance*
    columns derived from the raw *tournament* column.
    """
    df = df.copy()
    df["tournament_category"] = df["tournament"].apply(classify_tournament)
    df["tournament_code"] = df["tournament_category"].map(
        {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)}
    ).fillna(len(CATEGORY_ORDER) - 1).astype(int)
    df["tournament_importance"] = df["tournament_category"].map(CATEGORY_IMPORTANCE).fillna(
        CATEGORY_IMPORTANCE["other"]
    )
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, and days-since-epoch as numeric features."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    # Days since first match in dataset – used for temporal decay in XGBoost
    epoch = df["date"].min()
    df["days_from_epoch"] = (df["date"] - epoch).dt.days
    return df
