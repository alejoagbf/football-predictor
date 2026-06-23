"""
Match events estimator based on expected goals and team strength.

The international_results dataset has no corners, shots or card data,
so these statistics are modelled using well-established statistical
relationships from international football research.

Calibration constants are derived from international football averages:
  - Shots on target per team:  ~4.2 / game
  - Total shots per team:      ~11.5 / game
  - Corners per team:          ~5.4 / game
  - Yellow cards per team:     ~1.75 / game
  - Fouls per team:            ~13.0 / game
  - Red card prob per team:    ~7% / game

All event counts are modelled as Poisson-distributed, allowing us to
compute full over/under probabilities for each market.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson


# ── Calibration constants ────────────────────────────────────────────────────
# Based on analysis of international football (2010-2024):

AVG_LAMBDA = 1.35          # average xG per team per game

# Scaling ratios relative to expected goals
SHOTS_ON_TARGET_PER_LAMBDA = 3.11    # SoT = 3.11 * λ  → 4.2 when λ=1.35
TOTAL_SHOTS_PER_LAMBDA     = 8.53    # shots = 8.53 * λ → 11.5 when λ=1.35

# Corners: base rate + linear scaling with attack pressure
CORNERS_BASE               = 4.00
CORNERS_PER_LAMBDA         = 1.05    # corners = 4.0 + 1.05 * λ → 5.4 when λ=1.35

# Yellow cards: base + penalty for being the weaker team (more desperate fouls)
# YC = base + dominance_penalty * (opponent_λ / own_λ)
YC_BASE                    = 1.30
YC_DOMINANCE_FACTOR        = 0.35

# Fouls: similar logic
FOULS_BASE                 = 9.5
FOULS_DOMINANCE_FACTOR     = 2.6

# Red cards: low base rate, slightly elevated for weaker/desperate team
RED_BASE                   = 0.055
RED_DOMINANCE_FACTOR       = 0.018
RED_MAX                    = 0.22    # cap at 22%

# Offsides: correlated with attack depth
OFFSIDES_BASE              = 2.5
OFFSIDES_PER_LAMBDA        = 0.9


def _p_at_least_one(lam: float) -> float:
    """P(Poisson(lam) >= 1) = 1 - P(X=0)."""
    return float(1.0 - poisson.pmf(0, lam))


def _p_over(lam: float, threshold: float) -> float:
    """P(Poisson(lam) > threshold) using integer ceiling."""
    k = int(np.ceil(threshold))
    return float(1.0 - poisson.cdf(k - 1, lam))


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class MatchEventsEstimate:
    """Full estimated match statistics for both teams."""

    # ── Possession ────────────────────────────────────────────────────────────
    home_possession: float      # %
    away_possession: float      # %

    # ── Shots ─────────────────────────────────────────────────────────────────
    home_shots: float
    away_shots: float
    home_shots_on_target: float
    away_shots_on_target: float

    # ── Corners ───────────────────────────────────────────────────────────────
    home_corners: float
    away_corners: float
    total_corners: float

    # Corners over/under markets
    corners_over_8_5: float
    corners_over_9_5: float
    corners_over_10_5: float
    corners_over_11_5: float

    # ── Yellow cards ──────────────────────────────────────────────────────────
    home_yellow_cards: float
    away_yellow_cards: float
    total_yellow_cards: float

    # Cards over/under markets
    cards_over_2_5: float
    cards_over_3_5: float
    cards_over_4_5: float
    cards_over_5_5: float

    # ── Red cards ─────────────────────────────────────────────────────────────
    home_red_card_prob: float   # P(≥1 red card for home team)
    away_red_card_prob: float

    # ── Fouls ─────────────────────────────────────────────────────────────────
    home_fouls: float
    away_fouls: float
    total_fouls: float

    # ── Offsides ──────────────────────────────────────────────────────────────
    home_offsides: float
    away_offsides: float

    # ── Combined markets ──────────────────────────────────────────────────────
    home_clean_sheet_prob: float   # P(away goals = 0)
    away_clean_sheet_prob: float   # P(home goals = 0)


# ── Estimator ────────────────────────────────────────────────────────────────

def estimate_match_events(
    lambda_home: float,
    lambda_away: float,
    tournament_importance: float = 1.0,
) -> MatchEventsEstimate:
    """
    Estimate all match statistics given expected goals for each team.

    Parameters
    ----------
    lambda_home, lambda_away : float
        Expected goals from the ensemble model.
    tournament_importance : float
        Tournament weight (1=friendly, 4=World Cup). Higher importance
        slightly increases yellow cards due to greater match intensity.
    """
    lh = max(lambda_home, 0.05)
    la = max(lambda_away, 0.05)

    # ── Possession ────────────────────────────────────────────────────────────
    # Approximated from xG ratio (teams that create more also keep more ball)
    total_lambda = lh + la
    home_poss = round(lh / total_lambda * 100, 1)
    away_poss = round(100.0 - home_poss, 1)

    # ── Shots ─────────────────────────────────────────────────────────────────
    home_sot   = SHOTS_ON_TARGET_PER_LAMBDA * lh
    away_sot   = SHOTS_ON_TARGET_PER_LAMBDA * la
    home_shots = TOTAL_SHOTS_PER_LAMBDA * lh
    away_shots = TOTAL_SHOTS_PER_LAMBDA * la

    # ── Corners ───────────────────────────────────────────────────────────────
    home_corners = CORNERS_BASE + CORNERS_PER_LAMBDA * lh
    away_corners = CORNERS_BASE + CORNERS_PER_LAMBDA * la
    total_corners_lam = home_corners + away_corners

    # ── Yellow cards ──────────────────────────────────────────────────────────
    # Tournament importance boosts card rate (more pressure, less tolerance)
    importance_mult = 0.7 + 0.15 * tournament_importance

    home_yc_ratio = la / lh if lh > 0 else 1.0
    away_yc_ratio = lh / la if la > 0 else 1.0

    home_yc_lam = (YC_BASE + YC_DOMINANCE_FACTOR * home_yc_ratio) * importance_mult
    away_yc_lam = (YC_BASE + YC_DOMINANCE_FACTOR * away_yc_ratio) * importance_mult
    total_yc_lam = home_yc_lam + away_yc_lam

    # ── Red cards ─────────────────────────────────────────────────────────────
    home_red_lam = min(RED_BASE + RED_DOMINANCE_FACTOR * home_yc_ratio, RED_MAX)
    away_red_lam = min(RED_BASE + RED_DOMINANCE_FACTOR * away_yc_ratio, RED_MAX)

    # ── Fouls ─────────────────────────────────────────────────────────────────
    home_fouls_lam = FOULS_BASE + FOULS_DOMINANCE_FACTOR * home_yc_ratio
    away_fouls_lam = FOULS_BASE + FOULS_DOMINANCE_FACTOR * away_yc_ratio

    # ── Offsides ──────────────────────────────────────────────────────────────
    home_offsides_lam = OFFSIDES_BASE + OFFSIDES_PER_LAMBDA * lh
    away_offsides_lam = OFFSIDES_BASE + OFFSIDES_PER_LAMBDA * la

    # ── Clean sheet probabilities ─────────────────────────────────────────────
    home_cs = float(poisson.pmf(0, la))   # home keeps clean sheet = away scores 0
    away_cs = float(poisson.pmf(0, lh))   # away keeps clean sheet = home scores 0

    return MatchEventsEstimate(
        # Possession
        home_possession=home_poss,
        away_possession=away_poss,

        # Shots
        home_shots=round(home_shots, 1),
        away_shots=round(away_shots, 1),
        home_shots_on_target=round(home_sot, 1),
        away_shots_on_target=round(away_sot, 1),

        # Corners
        home_corners=round(home_corners, 1),
        away_corners=round(away_corners, 1),
        total_corners=round(home_corners + away_corners, 1),
        corners_over_8_5=round(_p_over(total_corners_lam, 8.5), 3),
        corners_over_9_5=round(_p_over(total_corners_lam, 9.5), 3),
        corners_over_10_5=round(_p_over(total_corners_lam, 10.5), 3),
        corners_over_11_5=round(_p_over(total_corners_lam, 11.5), 3),

        # Yellow cards
        home_yellow_cards=round(home_yc_lam, 2),
        away_yellow_cards=round(away_yc_lam, 2),
        total_yellow_cards=round(total_yc_lam, 2),
        cards_over_2_5=round(_p_over(total_yc_lam, 2.5), 3),
        cards_over_3_5=round(_p_over(total_yc_lam, 3.5), 3),
        cards_over_4_5=round(_p_over(total_yc_lam, 4.5), 3),
        cards_over_5_5=round(_p_over(total_yc_lam, 5.5), 3),

        # Red cards
        home_red_card_prob=round(_p_at_least_one(home_red_lam), 3),
        away_red_card_prob=round(_p_at_least_one(away_red_lam), 3),

        # Fouls
        home_fouls=round(home_fouls_lam, 1),
        away_fouls=round(away_fouls_lam, 1),
        total_fouls=round(home_fouls_lam + away_fouls_lam, 1),

        # Offsides
        home_offsides=round(home_offsides_lam, 1),
        away_offsides=round(away_offsides_lam, 1),

        # Clean sheets
        home_clean_sheet_prob=round(home_cs, 3),
        away_clean_sheet_prob=round(away_cs, 3),
    )
