# Football Match Predictor

A professional-grade international football prediction system combining a **Bayesian Hierarchical Poisson Model** and an **XGBoost ensemble** trained on 49,000+ historical international results (1872–present).

## What it predicts

| Output | Description |
|--------|-------------|
| `home_win / draw / away_win` | Match result probabilities |
| `expected_goals_home/away` | λ values from ensemble |
| `most_likely_score` | Single most probable exact score |
| `btts` | Both teams to score probability |
| `over_0.5 / 1.5 / 2.5 / 3.5` | Total goals markets |
| `score_probabilities` | Full distribution 0-0 to 7-7 |

## Architecture

```
Dataset (international_results)
        │
        ▼
┌───────────────────────────────────────────┐
│            Feature Pipeline               │
│  ELO (full history) + Rolling Form (5/10) │
│  Head-to-Head + Tournament Encoding       │
└───────────┬───────────────────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐   ┌──────────┐
│ Bayesian│   │ XGBoost  │
│Hierarch.│   │ Poisson  │
│  Poisson│   │Regressor │
│  (PyMC5)│   │ (Optuna) │
└────┬────┘   └────┬─────┘
     │λ_bayes      │λ_xgb
     └──────┬──────┘
            ▼
     ┌──────────────┐
     │   Ensemble   │  0.6·λ_bayes + 0.4·λ_xgb
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │Poisson Matrix│  0..7 × 0..7
     └──────┬───────┘
            ▼
     All market probabilities
```

### Model A — Bayesian Hierarchical Poisson

- Implemented in PyMC 5 with NUTS sampler
- Learns latent attack / defence strength per team
- Priors: `attack ~ N(0, σ_att)`, `defence ~ N(0, σ_def)`, `σ ~ HalfNormal(0.5)`
- Home advantage modelled as a trainable scalar (`home_adv ~ N(0.25, 0.1)`)
- **Temporal weighting** via `pm.Potential`: exponential decay amplifies recent matches
- Trained on the **last 10 years** of data for computational feasibility
- Predictions use **posterior mean parameters** for fast inference

### Model B — XGBoost Poisson Regressor

- Two independent models: one for home goals, one for away goals
- `objective = 'count:poisson'` — output is already a λ estimate
- **Optuna** hyperparameter search (TPE sampler, 80 trials by default)
- Trained with **temporal sample weights** on the full dataset
- Also includes a 3-class result classifier (H/D/A) for probability cross-checking

### Ensemble

Configurable weighted average (default: 60% Bayesian, 40% XGBoost):

```
λ_final = w_bayes · λ_bayes + w_xgb · λ_xgb
```

Weights can be overridden per-request via the API.

## Features engineered

| Feature group | Details |
|---------------|---------|
| **ELO** | Dynamic ratings updated after every match; K-factor by tournament; goal-diff multiplier |
| **Form (5 games)** | Wins/draws/losses, GF/GA/GD, win rate |
| **Form (10 games)** | Same stats over wider window |
| **Strength** | Rolling attack/defence averages (last 20 games) |
| **Head-to-head** | Last 10 H2H meetings; home wins, draws, away wins, avg goals |
| **Rest** | Days since each team's previous match |
| **Tournament** | 7-category encoding + importance weight |
| **Venue** | Binary neutral-ground indicator |
| **Temporal** | Year, month |

## Project structure

```
football_predictor/
├── data/
│   ├── raw/            # results.csv (downloaded automatically)
│   └── processed/      # features.parquet, pipeline.joblib
├── models/
│   ├── bayesian/       # trace.nc + meta.pkl
│   └── xgboost/        # model_home/away/result.joblib
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features/
│   │   ├── elo.py          # ELO system
│   │   ├── form.py         # Rolling form & H2H
│   │   ├── encoders.py     # Tournament encoding
│   │   └── pipeline.py     # Feature orchestration
│   ├── models/
│   │   ├── bayesian.py     # PyMC hierarchical model
│   │   ├── xgboost_model.py
│   │   └── ensemble.py
│   ├── training/
│   │   └── validation.py   # Walk-forward validation
│   ├── prediction/
│   │   ├── poisson.py      # Score matrix
│   │   └── predictor.py    # MatchPredictor
│   └── api/
│       ├── schemas.py
│       └── main.py
├── tests/
├── train.py
├── predict.py
└── requirements.txt
```

## Installation

```bash
cd football_predictor
pip install -r requirements.txt
```

## Training

```bash
# Full pipeline (downloads data, trains both models, ~30-90 min depending on hardware)
python train.py

# Skip Bayesian (fast, XGBoost only — ~5 min)
python train.py --skip-bayesian

# Skip Optuna (default XGBoost params — ~1 min)
python train.py --no-optimize

# Run walk-forward validation after training
python train.py --validate

# Custom ensemble (8-year Bayesian window, 100 Optuna trials)
python train.py --bayesian-years 8 --optuna-trials 100
```

## Prediction (CLI)

```bash
python predict.py "Argentina" "Brazil"
python predict.py "France" "Germany" --neutral --tournament "UEFA Euro"
python predict.py "Spain" "Portugal" --json
python predict.py "Brazil" "England" --weights 0.7 0.3
```

## Prediction (API)

```bash
uvicorn src.api.main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Argentina", "away_team": "Brazil"}'
```

Response:
```json
{
  "home_team": "Argentina",
  "away_team": "Brazil",
  "home_win": 0.4132,
  "draw": 0.2814,
  "away_win": 0.3054,
  "expected_goals_home": 1.85,
  "expected_goals_away": 1.33,
  "most_likely_score": "1-1",
  "btts": 0.6124,
  "over_0_5": 0.9541,
  "over_1_5": 0.7823,
  "over_2_5": 0.5612,
  "over_3_5": 0.3201,
  "score_probabilities": {"1-1": 0.0987, "2-1": 0.0832, ...},
  "lambda_bayes_home": 1.92,
  "lambda_bayes_away": 1.28,
  "lambda_xgb_home": 1.71,
  "lambda_xgb_away": 1.42,
  "model_weights": {"bayesian": 0.6, "xgboost": 0.4}
}
```

Additional endpoints:
- `GET /health` — model loading status
- `GET /teams` — list all known teams
- `GET /teams/{team_name}/strength` — Bayesian strength estimates

## Running tests

```bash
pytest tests/ -v
```

## Validation methodology

Walk-forward validation preserves temporal order:

| Fold | Train | Validate |
|------|-------|----------|
| 1 | 1990–2015 | 2016 |
| 2 | 1990–2016 | 2017 |
| … | … | … |
| N | 1990–2023 | 2024 |

Reported metrics per fold: Accuracy, Log-Loss, Brier Score (H/D/A), RMSE/MAE for goals.

## Design decisions & trade-offs

| Decision | Rationale |
|----------|-----------|
| Bayesian on last 10 years only | NUTS on 49k rows would take hours; recent data is more relevant |
| XGBoost on full dataset + temporal weights | Benefits from historical patterns; recency controlled by weight decay |
| `count:poisson` objective | Output is λ directly; no need for ReLU clipping of negative values |
| Independent Poisson per team | Dixon-Coles correlation correction not implemented (extension point) |
| Posterior mean for inference | Avoids full posterior predictive (slower); acceptable approximation |
| Walk-forward (not random k-fold) | Random split leaks future ELO/form signals into training set |

## Data source

[martj42/international_results](https://github.com/martj42/international_results) — 49,398 international football matches, CC0 licence.
