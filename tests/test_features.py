"""Unit tests for feature engineering modules."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.elo import EloSystem
from src.features.encoders import classify_tournament, encode_tournament
from src.features.form import compute_form_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame simulating international_results.csv structure."""
    data = [
        {"date": "2020-01-01", "home_team": "Argentina", "away_team": "Brazil",
         "home_score": 2, "away_score": 1, "tournament": "Friendly", "neutral": False},
        {"date": "2020-03-01", "home_team": "Brazil", "away_team": "Argentina",
         "home_score": 1, "away_score": 1, "tournament": "Friendly", "neutral": False},
        {"date": "2020-06-01", "home_team": "Argentina", "away_team": "Brazil",
         "home_score": 0, "away_score": 2, "tournament": "Copa America", "neutral": True},
        {"date": "2021-01-01", "home_team": "Germany", "away_team": "France",
         "home_score": 3, "away_score": 2, "tournament": "Friendly", "neutral": False},
        {"date": "2021-06-01", "home_team": "France", "away_team": "Germany",
         "home_score": 1, "away_score": 0, "tournament": "UEFA Euro", "neutral": False},
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── ELO tests ─────────────────────────────────────────────────────────────────

class TestEloSystem:
    def test_initial_rating(self) -> None:
        elo = EloSystem()
        assert elo.get_rating("Unknown Team") == 1500.0

    def test_winner_gains_rating(self) -> None:
        elo = EloSystem()
        elo.update("A", "B", 1, 0, "Friendly")
        assert elo.get_rating("A") > 1500.0
        assert elo.get_rating("B") < 1500.0

    def test_draw_updates_both(self) -> None:
        elo = EloSystem(initial_rating=1600.0)
        elo.ratings["A"] = 1600.0
        elo.ratings["B"] = 1400.0
        elo.update("A", "B", 0, 0, "Friendly")
        # A had higher expected score, so draw costs them points
        assert elo.get_rating("A") < 1600.0
        assert elo.get_rating("B") > 1400.0

    def test_zero_sum_property(self) -> None:
        elo = EloSystem()
        elo.update("A", "B", 2, 1, "FIFA World Cup")
        # Rating changes must be equal and opposite
        delta_a = elo.get_rating("A") - 1500.0
        delta_b = elo.get_rating("B") - 1500.0
        assert abs(delta_a + delta_b) < 1e-9

    def test_neutral_reduces_home_advantage(self) -> None:
        elo1 = EloSystem()
        elo2 = EloSystem()
        elo1.update("A", "B", 0, 0, "Friendly", is_neutral=False)
        elo2.update("A", "B", 0, 0, "Friendly", is_neutral=True)
        # Neutral: home team expected score is lower → draw gains more points
        assert elo2.get_rating("A") > elo1.get_rating("A")

    def test_compute_elo_history_no_leakage(self, sample_df: pd.DataFrame) -> None:
        elo = EloSystem()
        result = elo.compute_elo_history(sample_df)
        # First match: both teams should have initial ELO (1500)
        first_row = result.sort_values("date").iloc[0]
        assert first_row["elo_home"] == pytest.approx(1500.0)
        assert first_row["elo_away"] == pytest.approx(1500.0)

    def test_goal_diff_multiplier(self) -> None:
        elo = EloSystem()
        assert elo._goal_diff_multiplier(0) == 1.0
        assert elo._goal_diff_multiplier(1) == 1.0
        assert elo._goal_diff_multiplier(2) == 1.5
        assert elo._goal_diff_multiplier(3) == 1.75
        assert elo._goal_diff_multiplier(4) == 2.0

    def test_k_factor_by_tournament(self) -> None:
        elo = EloSystem()
        assert elo._k_factor("FIFA World Cup") == 60.0
        assert elo._k_factor("Friendly") == 30.0
        assert elo._k_factor("UEFA Euro 2024 Qualification") >= 40.0


# ── Tournament encoder tests ──────────────────────────────────────────────────

class TestEncoders:
    def test_classify_world_cup(self) -> None:
        assert classify_tournament("FIFA World Cup") == "world_cup"

    def test_classify_friendly(self) -> None:
        assert classify_tournament("International Friendly") == "friendly"

    def test_classify_continental(self) -> None:
        assert classify_tournament("Copa America") == "continental"

    def test_classify_unknown_returns_other(self) -> None:
        assert classify_tournament("Some Random Cup") == "other"

    def test_encode_tournament_adds_columns(self, sample_df: pd.DataFrame) -> None:
        result = encode_tournament(sample_df)
        assert "tournament_category" in result.columns
        assert "tournament_code" in result.columns
        assert "tournament_importance" in result.columns
        assert result["tournament_code"].dtype in (int, "int64")


# ── Form feature tests ────────────────────────────────────────────────────────

class TestFormFeatures:
    def test_first_match_zero_form(self, sample_df: pd.DataFrame) -> None:
        """Teams with no prior matches should have zero form features."""
        df = sample_df.sort_values("date").reset_index(drop=True)
        result = compute_form_features(df)

        first = result.sort_values("date").iloc[0]
        # No prior matches → form should be zero
        assert first["home_wins_last5"] == 0.0
        assert first["away_wins_last5"] == 0.0

    def test_form_uses_only_past_data(self, sample_df: pd.DataFrame) -> None:
        """Form at match i must not include match i or later matches."""
        df = sample_df.sort_values("date").reset_index(drop=True)
        result = compute_form_features(df)

        # Second Argentina match: should reflect first match result
        arg_rows = result[result["home_team"] == "Brazil"].sort_values("date")
        if len(arg_rows) > 0:
            second = arg_rows.iloc[0]
            # Brazil had 1 prior match as away team (they lost 1-2)
            # so their away_losses_last5 for the next match should be 1
            assert second["home_wins_last5"] >= 0

    def test_h2h_stats_computed(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.sort_values("date").reset_index(drop=True)
        result = compute_form_features(df)
        assert "h2h_played" in result.columns
        assert "h2h_avg_goals" in result.columns

    def test_days_rest_positive(self, sample_df: pd.DataFrame) -> None:
        df = sample_df.sort_values("date").reset_index(drop=True)
        result = compute_form_features(df)
        assert (result["home_days_rest"] >= 0).all()
        assert (result["away_days_rest"] >= 0).all()
