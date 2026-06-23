"""Pydantic v2 schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    home_team: str = Field(..., examples=["Argentina"], description="Home team name")
    away_team: str = Field(..., examples=["Brazil"], description="Away team name")
    is_neutral: bool = Field(False, description="True if match is at a neutral venue")
    tournament: str = Field("Friendly", description="Tournament name (affects features)")
    weight_bayes: float = Field(
        0.6, ge=0.0, le=1.0, description="Bayesian model weight (must sum to 1 with weight_xgb)"
    )
    weight_xgb: float = Field(
        0.4, ge=0.0, le=1.0, description="XGBoost model weight"
    )

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "PredictRequest":
        total = self.weight_bayes + self.weight_xgb
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"weight_bayes + weight_xgb must equal 1.0 (got {total:.4f})"
            )
        return self


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    home_team: str
    away_team: str

    # Match result probabilities
    home_win: float = Field(..., description="P(home team wins)")
    draw: float = Field(..., description="P(draw)")
    away_win: float = Field(..., description="P(away team wins)")

    # Expected goals
    expected_goals_home: float
    expected_goals_away: float

    # Most likely exact score  (e.g. "1-0")
    most_likely_score: str

    # Markets
    btts: float = Field(..., description="P(both teams score)")
    over_0_5: float
    over_1_5: float
    over_2_5: float
    over_3_5: float

    # Score distribution
    score_probabilities: dict[str, float] = Field(
        ..., description="Probability for each score (filtered to >0.1%)"
    )

    # Internal decomposition (useful for debugging / transparency)
    lambda_bayes_home: float
    lambda_bayes_away: float
    lambda_xgb_home: float
    lambda_xgb_away: float
    model_weights: dict[str, float]


class TeamStrengthResponse(BaseModel):
    """Response for GET /teams/{team_name}/strength."""

    team: str
    elo_rating: float
    attack_posterior_mean: float
    defence_posterior_mean: float
    net_strength: float


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    models_loaded: bool
    bayesian_train_cutoff: str | None = None
