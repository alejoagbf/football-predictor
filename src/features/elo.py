"""
ELO Rating System for International Football.

Implements the World Football ELO standard (https://www.eloratings.net/about):
  - Team ratings start at 1500
  - K-factor scales with tournament importance and goal difference
  - Home advantage is modelled as a 100-point rating bonus in expected-score calc
  - For neutral venues the bonus is not applied
  - Ratings are updated AFTER each match to guarantee zero data leakage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import pandas as pd

from src.config import (
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL_RATING,
    ELO_K_DEFAULT,
    ELO_K_FACTORS,
)

logger = logging.getLogger(__name__)


@dataclass
class EloSystem:
    """Chronological ELO calculator for international football."""

    initial_rating: float = ELO_INITIAL_RATING
    ratings: dict[str, float] = field(default_factory=dict)

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_rating(self, team: str) -> float:
        """Return current rating for *team*, defaulting to initial_rating."""
        return self.ratings.get(team, self.initial_rating)

    def expected_score(
        self,
        home_elo: float,
        away_elo: float,
        is_neutral: bool = False,
    ) -> float:
        """
        Probability that the home side wins (or halves in a draw).

        For non-neutral venues the home team receives ELO_HOME_ADVANTAGE
        extra points before computing the logistic expectation.
        """
        adj = home_elo if is_neutral else home_elo + ELO_HOME_ADVANTAGE
        return 1.0 / (1.0 + 10.0 ** ((away_elo - adj) / 400.0))

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str,
        is_neutral: bool = False,
    ) -> Tuple[float, float]:
        """
        Compute updated ratings after a match result.

        Call this AFTER capturing pre-match ratings to avoid data leakage.
        Returns (new_home_elo, new_away_elo).
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        expected_home = self.expected_score(home_elo, away_elo, is_neutral)

        # Actual outcome from home perspective
        if home_score > away_score:
            actual = 1.0
        elif home_score == away_score:
            actual = 0.5
        else:
            actual = 0.0

        k = self._k_factor(tournament)
        mult = self._goal_diff_multiplier(abs(home_score - away_score))
        delta = k * mult * (actual - expected_home)

        self.ratings[home_team] = home_elo + delta
        self.ratings[away_team] = away_elo - delta

        return self.ratings[home_team], self.ratings[away_team]

    # ── Core computation ──────────────────────────────────────────────────────

    def compute_elo_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add *elo_home*, *elo_away*, *elo_diff* columns to *df*.

        Pre-match ELO values are recorded before each update so there is
        no leakage.  The DataFrame must be sorted by date ascending (the
        function re-sorts internally to be safe).
        """
        df = df.sort_values("date").reset_index(drop=True).copy()

        pre_home: list[float] = []
        pre_away: list[float] = []

        for _, row in df.iterrows():
            h = str(row["home_team"])
            a = str(row["away_team"])

            # Record BEFORE update
            pre_home.append(self.get_rating(h))
            pre_away.append(self.get_rating(a))

            # Update ratings with match outcome
            self.update(
                home_team=h,
                away_team=a,
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                tournament=str(row["tournament"]),
                is_neutral=bool(row.get("neutral", False)),
            )

        df["elo_home"] = pre_home
        df["elo_away"] = pre_away
        df["elo_diff"] = df["elo_home"] - df["elo_away"]

        logger.info("Computed ELO history for %d matches.", len(df))
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    def _k_factor(self, tournament: str) -> float:
        t_lower = tournament.lower()
        for name, k in ELO_K_FACTORS.items():
            if name.lower() in t_lower:
                return k
        return ELO_K_DEFAULT

    @staticmethod
    def _goal_diff_multiplier(goal_diff: int) -> float:
        """
        Amplify rating change for emphatic victories.

        World ELO formula:
          1 goal  → 1.00
          2 goals → 1.50
          3+ goals → 1.75 + 0.25*(n-3)
        """
        if goal_diff <= 1:
            return 1.0
        if goal_diff == 2:
            return 1.5
        return 1.75 + 0.25 * (goal_diff - 3)
