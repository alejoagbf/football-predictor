"""
Bayesian Hierarchical Poisson Model for goal prediction.

Architecture
------------
The model learns a latent attack strength and defence strength for every
national team.  Expected goals follow a log-linear Poisson model:

    log(λ_home) = home_adv  +  attack[home]  −  defence[away]
    log(λ_away) =              attack[away]  −  defence[home]

Priors
------
    home_adv  ~ Normal(0.25, 0.1)
    σ_att     ~ HalfNormal(0.5)
    σ_def     ~ HalfNormal(0.5)
    attack[i] ~ Normal(0, σ_att)   for every team i
    defence[i]~ Normal(0, σ_def)   for every team i

Temporal weighting
------------------
Recent matches receive higher log-likelihood weight via pm.Potential,
implementing exponential decay without modifying the likelihood structure.

Training scope
--------------
Only the last BAYESIAN_DATA_YEARS of data are used so that NUTS stays
tractable.  The ELO and form features are computed on the full dataset
but the Bayesian posterior is conditioned on recent evidence only.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.config import (
    BAYESIAN_CHAINS,
    BAYESIAN_DATA_YEARS,
    BAYESIAN_MODEL_DIR,
    BAYESIAN_RANDOM_SEED,
    BAYESIAN_SAMPLES,
    BAYESIAN_TARGET_ACCEPT,
    BAYESIAN_TUNE,
    XGBOOST_TEMPORAL_DECAY,
)

logger = logging.getLogger(__name__)

_TRACE_FILE = "trace.nc"
_META_FILE = "meta.pkl"


@dataclass
class BayesianModelMeta:
    """Serialisable metadata alongside the ArviZ InferenceData trace."""

    teams: list[str]
    team_to_idx: dict[str, int]
    posterior_means: dict[str, np.ndarray]  # attack, defence, home_adv
    n_teams: int
    train_cutoff: pd.Timestamp


class BayesianHierarchicalModel:
    """
    Wrapper around a PyMC hierarchical Poisson model.

    After fitting, predictions are made via posterior mean parameters
    (a fast approximation that avoids running MCMC at inference time).
    For full posterior predictive inference, use *sample_posterior_predictive*.
    """

    def __init__(self) -> None:
        self.meta: Optional[BayesianModelMeta] = None
        self.trace: Optional[az.InferenceData] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "BayesianHierarchicalModel":
        """
        Train the model on *df*.

        Only rows within the last BAYESIAN_DATA_YEARS are used for the
        Bayesian likelihood; temporal decay weights amplify recent matches.
        """
        # Filter to recent data
        cutoff_date = df["date"].max() - pd.DateOffset(years=BAYESIAN_DATA_YEARS)
        recent = df[df["date"] >= cutoff_date].copy()
        logger.info(
            "Bayesian training on %d matches (%s – %s)",
            len(recent),
            recent["date"].min().date(),
            recent["date"].max().date(),
        )

        # Index teams
        all_teams = sorted(
            set(recent["home_team"].tolist() + recent["away_team"].tolist())
        )
        team_to_idx = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)

        home_idx = recent["home_team"].map(team_to_idx).values.astype(int)
        away_idx = recent["away_team"].map(team_to_idx).values.astype(int)
        home_goals = recent["home_score"].values.astype(int)
        away_goals = recent["away_score"].values.astype(int)
        is_neutral = recent["neutral"].astype(int).values

        # Temporal decay weights (normalised to mean=1)
        days_from_max = (recent["date"].max() - recent["date"]).dt.days.values
        raw_weights = np.exp(-XGBOOST_TEMPORAL_DECAY * days_from_max)
        weights = raw_weights / raw_weights.mean()

        logger.info("Building PyMC model with %d teams …", n_teams)
        with pm.Model() as model:
            # ── Hyperpriors ───────────────────────────────────────────────────
            sigma_att = pm.HalfNormal("sigma_att", sigma=0.5)
            sigma_def = pm.HalfNormal("sigma_def", sigma=0.5)

            # Home advantage (reduced to 0 for neutral venues)
            home_adv_raw = pm.Normal("home_adv", mu=0.25, sigma=0.1)
            home_adv = home_adv_raw * (1.0 - is_neutral)

            # ── Team parameters ───────────────────────────────────────────────
            attack = pm.Normal("attack", mu=0.0, sigma=sigma_att, shape=n_teams)
            defence = pm.Normal("defence", mu=0.0, sigma=sigma_def, shape=n_teams)

            # ── Expected goals (log-linear Poisson) ───────────────────────────
            log_lambda_home = home_adv + attack[home_idx] - defence[away_idx]
            log_lambda_away = attack[away_idx] - defence[home_idx]
            lambda_home = pm.math.exp(log_lambda_home)
            lambda_away = pm.math.exp(log_lambda_away)

            # ── Weighted log-likelihood via Potential ─────────────────────────
            # This implements temporal decay: recent matches carry more weight.
            w = pt.as_tensor_variable(weights)
            ll_home = w * pm.logp(pm.Poisson.dist(mu=lambda_home), home_goals)
            ll_away = w * pm.logp(pm.Poisson.dist(mu=lambda_away), away_goals)
            pm.Potential("weighted_ll_home", ll_home.sum())
            pm.Potential("weighted_ll_away", ll_away.sum())

            # ── MCMC sampling ─────────────────────────────────────────────────
            logger.info(
                "Starting NUTS sampler (chains=%d, draws=%d, tune=%d) …",
                BAYESIAN_CHAINS,
                BAYESIAN_SAMPLES,
                BAYESIAN_TUNE,
            )
            # PyMC 6: target_accept is passed via nuts={} dict
            trace = pm.sample(
                draws=BAYESIAN_SAMPLES,
                tune=BAYESIAN_TUNE,
                chains=BAYESIAN_CHAINS,
                nuts={"target_accept": BAYESIAN_TARGET_ACCEPT},
                random_seed=BAYESIAN_RANDOM_SEED,
                progressbar=True,
                return_inferencedata=True,
            )

        # Extract posterior means for fast inference
        posterior_means = {
            "attack": trace.posterior["attack"].values.mean(axis=(0, 1)),
            "defence": trace.posterior["defence"].values.mean(axis=(0, 1)),
            "home_adv": float(trace.posterior["home_adv"].values.mean()),
        }

        self.trace = trace
        self.meta = BayesianModelMeta(
            teams=all_teams,
            team_to_idx=team_to_idx,
            posterior_means=posterior_means,
            n_teams=n_teams,
            train_cutoff=recent["date"].max(),
        )
        logger.info("Bayesian model fitted successfully.")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_lambdas(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = False,
    ) -> tuple[float, float]:
        """
        Predict (lambda_home, lambda_away) using posterior mean parameters.

        Unknown teams receive the global mean (attack=0, defence=0).
        """
        if self.meta is None:
            raise RuntimeError("Model has not been fitted yet.")

        pm_ = self.meta.posterior_means
        t2i = self.meta.team_to_idx

        att_home = pm_["attack"][t2i[home_team]] if home_team in t2i else 0.0
        def_home = pm_["defence"][t2i[home_team]] if home_team in t2i else 0.0
        att_away = pm_["attack"][t2i[away_team]] if away_team in t2i else 0.0
        def_away = pm_["defence"][t2i[away_team]] if away_team in t2i else 0.0

        home_adv = 0.0 if is_neutral else pm_["home_adv"]

        lambda_home = float(np.exp(home_adv + att_home - def_away))
        lambda_away = float(np.exp(att_away - def_home))

        return lambda_home, lambda_away

    def team_strengths(self) -> pd.DataFrame:
        """Return a DataFrame with attack/defence posterior means per team."""
        if self.meta is None:
            raise RuntimeError("Model not fitted.")
        pm_ = self.meta.posterior_means
        return pd.DataFrame(
            {
                "team": self.meta.teams,
                "attack": pm_["attack"],
                "defence": pm_["defence"],
                "net": pm_["attack"] - pm_["defence"],
            }
        ).sort_values("net", ascending=False)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path | None = None) -> Path:
        out = directory or BAYESIAN_MODEL_DIR
        out.mkdir(parents=True, exist_ok=True)

        self.trace.to_netcdf(str(out / _TRACE_FILE))  # type: ignore[union-attr]
        with open(out / _META_FILE, "wb") as f:
            pickle.dump(self.meta, f)

        logger.info("Saved Bayesian model to %s", out)
        return out

    @classmethod
    def load(cls, directory: Path | None = None) -> "BayesianHierarchicalModel":
        d = directory or BAYESIAN_MODEL_DIR
        obj = cls()
        obj.trace = az.from_netcdf(str(d / _TRACE_FILE))
        with open(d / _META_FILE, "rb") as f:
            obj.meta = pickle.load(f)
        logger.info("Loaded Bayesian model from %s", d)
        return obj
