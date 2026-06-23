"""
FastAPI application for football match predictions.

Endpoints
---------
GET  /health                         — health check
POST /predict                        — full match prediction
GET  /teams/{team_name}/strength     — Bayesian team strength
GET  /teams                          — list all known teams

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TeamStrengthResponse,
)
from src.config import LOG_DATE_FORMAT, LOG_FORMAT
from src.prediction.predictor import MatchPredictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Global predictor (loaded once at startup) ─────────────────────────────────
_predictor: MatchPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all models once when the server starts."""
    global _predictor
    logger.info("Loading prediction models …")
    try:
        _predictor = MatchPredictor.load()
        logger.info("Models loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        _predictor = None
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Football Match Predictor",
    description=(
        "Bayesian + XGBoost ensemble for international football match prediction. "
        "Trained on 49k+ international results (1872–present)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Return API health status and model loading state."""
    loaded = _predictor is not None
    cutoff = None
    if loaded and _predictor.bayesian is not None and _predictor.bayesian.meta is not None:  # type: ignore[union-attr]
        cutoff = str(_predictor.bayesian.meta.train_cutoff.date())  # type: ignore[union-attr]
    return HealthResponse(
        status="ok" if loaded else "degraded",
        models_loaded=loaded,
        bayesian_train_cutoff=cutoff,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Generate a full probability prediction for an upcoming match.

    The response includes 1X2 probabilities, expected goals, most likely
    score, BTTS, over/under markets, and the full score distribution.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded.  Run train.py first.")

    # Apply custom weights if provided
    _predictor.ensemble.set_weights(request.weight_bayes, request.weight_xgb)

    try:
        pred = _predictor.predict(
            home_team=request.home_team,
            away_team=request.away_team,
            is_neutral=request.is_neutral,
            tournament=request.tournament,
        )
    except Exception as exc:
        logger.exception("Prediction failed for %s vs %s", request.home_team, request.away_team)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(**pred.to_dict())


@app.get("/teams", response_model=list[str], tags=["Teams"])
def list_teams() -> list[str]:
    """Return all team names known to the Bayesian model."""
    if _predictor is None or _predictor.bayesian is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    if _predictor.bayesian.meta is None:
        raise HTTPException(status_code=503, detail="Bayesian model not trained.")
    return sorted(_predictor.bayesian.meta.teams)


@app.get("/teams/{team_name}/strength", response_model=TeamStrengthResponse, tags=["Teams"])
def team_strength(team_name: str) -> TeamStrengthResponse:
    """Return the Bayesian posterior strength estimates for a single team."""
    if _predictor is None or _predictor.bayesian is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    meta = _predictor.bayesian.meta
    if meta is None:
        raise HTTPException(status_code=503, detail="Bayesian model not trained.")

    if team_name not in meta.team_to_idx:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found.")

    idx = meta.team_to_idx[team_name]
    att = float(meta.posterior_means["attack"][idx])
    defs = float(meta.posterior_means["defence"][idx])
    elo = _predictor.pipeline.elo_system.get_rating(team_name) if _predictor.pipeline else 1500.0  # type: ignore[union-attr]

    return TeamStrengthResponse(
        team=team_name,
        elo_rating=round(elo, 1),
        attack_posterior_mean=round(att, 4),
        defence_posterior_mean=round(defs, 4),
        net_strength=round(att - defs, 4),
    )
