"""
Rolling form and strength features computed without data leakage.

Vectorised implementation: instead of iterating row-by-row (O(n²)),
we build a team-centric long-format DataFrame and use pandas rolling
operations, reducing complexity to O(n log n).

All feature values are computed using only matches that took place
*strictly before* the match date — guaranteed by a `.shift(1)` before
every rolling window, so the current match never influences its own features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import ATTACK_DEFENSE_WINDOW, FORM_WINDOWS, H2H_WINDOW

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_team_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the match-level DataFrame into a team-centric long format.

    Each match generates two rows (one for home team, one for away team)
    with normalised 'scored', 'conceded', 'won', 'drew', 'lost' columns
    from each team's perspective.
    """
    home = df[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent"})
    home["scored"] = home["home_score"]
    home["conceded"] = home["away_score"]
    home["won"] = (home["home_score"] > home["away_score"]).astype(float)
    home["drew"] = (home["home_score"] == home["away_score"]).astype(float)
    home["lost"] = (home["home_score"] < home["away_score"]).astype(float)

    away = df[["date", "away_team", "home_team", "away_score", "home_score"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent"})
    away["scored"] = away["away_score"]
    away["conceded"] = away["home_score"]
    away["won"] = (away["away_score"] > away["home_score"]).astype(float)
    away["drew"] = (away["away_score"] == away["home_score"]).astype(float)
    away["lost"] = (away["away_score"] < away["home_score"]).astype(float)

    long_df = pd.concat(
        [
            home[["date", "team", "opponent", "scored", "conceded", "won", "drew", "lost"]],
            away[["date", "team", "opponent", "scored", "conceded", "won", "drew", "lost"]],
        ],
        ignore_index=True,
    ).sort_values(["team", "date"]).reset_index(drop=True)

    return long_df


def _rolling_team_stats(long_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling stats per team using pandas rolling.

    Uses .shift(1) so match i only sees matches 0…i-1 (no leakage).
    Returns a DataFrame indexed by (team, date) with stat columns.
    """
    grp = long_df.groupby("team", sort=False)

    stats = (
        grp[["scored", "conceded", "won", "drew", "lost"]]
        .apply(
            lambda g: g.shift(1)              # exclude current match
             .rolling(window, min_periods=1)
             .agg(
                 matches=("scored", "count"),
                 wins=("won", "sum"),
                 draws=("drew", "sum"),
                 losses=("lost", "sum"),
                 gf=("scored", "sum"),
                 ga=("conceded", "sum"),
                 avg_gf=("scored", "mean"),
                 avg_ga=("conceded", "mean"),
                 win_rate=("won", "mean"),
             )
        )
    )

    # Older pandas: apply with include_groups deprecation
    # Flatten multi-index if present
    if isinstance(stats.index, pd.MultiIndex) and stats.index.names[0] == "team":
        stats = stats.reset_index(level=0, drop=True)

    stats["gd"] = stats["gf"] - stats["ga"]

    # For first match of each team (no prior data), replace NaN with 0
    stats = stats.fillna(0.0)

    # Attach team and date for later join
    stats["team"] = long_df["team"].values
    stats["date"] = long_df["date"].values
    return stats


def _compute_days_rest(long_df: pd.DataFrame) -> pd.DataFrame:
    """Days since previous match per team (no leakage via shift)."""
    grp = long_df.groupby("team", sort=False)

    def days_since_prev(g: pd.DataFrame) -> pd.Series:
        dates = g["date"]
        shifted = dates.shift(1)
        diff = (dates - shifted).dt.days.fillna(365.0)
        return diff

    rest = grp.apply(days_since_prev, include_groups=False).reset_index(level=0, drop=True)
    return pd.DataFrame({"team": long_df["team"].values, "date": long_df["date"].values, "days_rest": rest.values})


def _strength_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Attack/defence strength over the longer window (no leakage)."""
    grp = long_df.groupby("team", sort=False)
    stats = (
        grp[["scored", "conceded"]]
        .apply(
            lambda g: g.shift(1)
             .rolling(ATTACK_DEFENSE_WINDOW, min_periods=1)
             .mean()
        )
    )
    if isinstance(stats.index, pd.MultiIndex) and stats.index.names[0] == "team":
        stats = stats.reset_index(level=0, drop=True)
    stats = stats.fillna(0.0)
    stats.columns = ["attack_strength", "defense_strength"]
    stats["team"] = long_df["team"].values
    stats["date"] = long_df["date"].values
    return stats


def _h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Head-to-head statistics for each match (last H2H_WINDOW meetings).

    We iterate over unique team pairs only (much smaller than all matches),
    then join back to the main DataFrame.
    """
    # Create a pair key that is order-independent
    df = df.copy()
    df["pair"] = df.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )

    h2h_records: list[dict] = []
    for pair, group in df.groupby("pair"):
        group = group.sort_values("date").reset_index(drop=True)
        for i, row in group.iterrows():
            past = group[group["date"] < row["date"]].tail(H2H_WINDOW)
            n = len(past)
            if n == 0:
                h2h_records.append({
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "h2h_played": 0.0,
                    "h2h_home_wins": 0.0,
                    "h2h_draws": 0.0,
                    "h2h_away_wins": 0.0,
                    "h2h_avg_goals": 0.0,
                })
                continue

            home = row["home_team"]
            home_wins = (
                ((past["home_team"] == home) & (past["home_score"] > past["away_score"]))
                | ((past["away_team"] == home) & (past["away_score"] > past["home_score"]))
            ).sum()
            away_wins = (
                ((past["home_team"] == row["away_team"]) & (past["home_score"] > past["away_score"]))
                | ((past["away_team"] == row["away_team"]) & (past["away_score"] > past["home_score"]))
            ).sum()
            draws = n - home_wins - away_wins
            avg_goals = float((past["home_score"] + past["away_score"]).mean())

            h2h_records.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "h2h_played": float(n),
                "h2h_home_wins": float(home_wins),
                "h2h_draws": float(draws),
                "h2h_away_wins": float(away_wins),
                "h2h_avg_goals": avg_goals,
            })

    return pd.DataFrame(h2h_records)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all rolling form features for every match in *df*.

    Uses vectorised pandas rolling for O(n log n) complexity instead of
    the naive O(n²) row-iteration approach.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    n_matches = len(df)
    logger.info("Computing form features for %d matches (vectorised) ...", n_matches)

    # Build team-centric long format once
    long_df = _build_team_long(df)

    # Rolling stats for each window
    all_window_stats: dict[int, pd.DataFrame] = {}
    for w in FORM_WINDOWS:
        logger.info("  Rolling window %d ...", w)
        all_window_stats[w] = _rolling_team_stats(long_df, w)

    # Attack/defence strength
    logger.info("  Strength features ...")
    strength_df = _strength_stats(long_df)

    # Days rest
    logger.info("  Rest days ...")
    rest_df = _compute_days_rest(long_df)

    # H2H (still pair-wise but vastly fewer pairs than matches)
    logger.info("  Head-to-head (per pair) ...")
    h2h_df = _h2h_features(df)

    # ── Join everything back to the match-level DataFrame ─────────────────────
    # Build a (team, date) indexed lookup for each stat table
    def make_lookup(stat_df: pd.DataFrame) -> dict[tuple, dict]:
        return {
            (row["team"], row["date"]): row.drop(["team", "date"]).to_dict()
            for _, row in stat_df.iterrows()
        }

    lookups = {w: make_lookup(s) for w, s in all_window_stats.items()}
    strength_lookup = make_lookup(strength_df)
    rest_lookup = make_lookup(rest_df)

    records: list[dict] = []
    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]
        feat: dict = {}

        for prefix, team in [("home", home), ("away", away)]:
            for w in FORM_WINDOWS:
                stats = lookups[w].get((team, date), {})
                for k, v in stats.items():
                    feat[f"{prefix}_{k}_last{w}"] = v

            stre = strength_lookup.get((team, date), {})
            feat[f"{prefix}_attack_strength"] = stre.get("attack_strength", 0.0)
            feat[f"{prefix}_defense_strength"] = stre.get("defense_strength", 0.0)

            rest = rest_lookup.get((team, date), {})
            feat[f"{prefix}_days_rest"] = rest.get("days_rest", 365.0)

        records.append(feat)

    feat_df = pd.DataFrame(records, index=df.index)

    # Merge H2H — deduplicate first to avoid row inflation on multi-index join
    h2h_cols = ["h2h_played", "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_avg_goals"]
    h2h_dedup = h2h_df.drop_duplicates(subset=["date", "home_team", "away_team"])
    df = df.merge(
        h2h_dedup[["date", "home_team", "away_team"] + h2h_cols],
        on=["date", "home_team", "away_team"],
        how="left",
    )
    for c in h2h_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    result = pd.concat([df, feat_df], axis=1)
    logger.info(
        "Form features done. Shape: %d rows x %d cols",
        len(result),
        len(result.columns),
    )
    return result
