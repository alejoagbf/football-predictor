"""
Master training script.

Usage
-----
    python train.py                        # Full pipeline
    python train.py --skip-bayesian        # XGBoost only (fast)
    python train.py --validate             # Add walk-forward validation
    python train.py --no-optimize          # Skip Optuna (use default XGB params)
    python train.py --bayesian-years 8     # Use last N years for Bayesian
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.config import (
    BAYESIAN_DATA_YEARS,
    DATA_PROCESSED_DIR,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    VALIDATION_START_YEAR,
    VALIDATION_TRAIN_START_YEAR,
    XGBOOST_OPTUNA_TRIALS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the football prediction system.")
    p.add_argument("--skip-bayesian", action="store_true", help="Skip Bayesian model (much faster)")
    p.add_argument("--no-optimize", action="store_true", help="Skip Optuna hyperparameter search")
    p.add_argument("--validate", action="store_true", help="Run walk-forward validation after training")
    p.add_argument("--optuna-trials", type=int, default=XGBOOST_OPTUNA_TRIALS)
    p.add_argument("--bayesian-years", type=int, default=BAYESIAN_DATA_YEARS)
    p.add_argument("--force-download", action="store_true", help="Re-download dataset")
    p.add_argument("--val-start-year", type=int, default=VALIDATION_START_YEAR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.time()

    # ── 1. Download / load data ───────────────────────────────────────────────
    logger.info("-- Step 1/5: Loading dataset --")
    from src.data_loader import download_dataset, load_results
    download_dataset(force=args.force_download)
    df_raw = load_results()
    logger.info("Loaded %d matches.", len(df_raw))

    # ── 2. Feature engineering ────────────────────────────────────────────────
    logger.info("-- Step 2/5: Feature engineering --")
    from src.features.pipeline import FeaturePipeline, save_features
    t0 = time.time()
    pipeline = FeaturePipeline()
    df_features = pipeline.fit_transform(df_raw)
    save_features(df_features)
    pipeline.save()
    logger.info("Features done in %.1f min", (time.time() - t0) / 60)

    # ── 3. Bayesian model ─────────────────────────────────────────────────────
    if not args.skip_bayesian:
        logger.info("-- Step 3/5: Bayesian hierarchical model --")
        import src.config as cfg
        cfg.BAYESIAN_DATA_YEARS = args.bayesian_years  # allow CLI override
        from src.models.bayesian import BayesianHierarchicalModel
        t0 = time.time()
        bayes = BayesianHierarchicalModel()
        bayes.fit(df_features)
        bayes.save()
        logger.info("Bayesian done in %.1f min", (time.time() - t0) / 60)

        strengths = bayes.team_strengths().head(20)
        logger.info("Top-20 teams by net strength:\n%s", strengths.to_string(index=False))
    else:
        logger.info("-- Step 3/5: Bayesian model SKIPPED --")

    # ── 4. XGBoost model ─────────────────────────────────────────────────────
    logger.info("-- Step 4/5: XGBoost model --")
    from src.models.xgboost_model import XGBoostPredictor
    t0 = time.time()
    xgb_model = XGBoostPredictor()
    xgb_model.fit(
        df_features,
        optimize=not args.no_optimize,
        n_trials=args.optuna_trials,
    )
    xgb_model.save()
    logger.info("XGBoost done in %.1f min", (time.time() - t0) / 60)

    importances = xgb_model.feature_importances().head(15)
    logger.info("Top-15 features:\n%s", importances.to_string(index=False))

    # ── 5. Walk-forward validation ────────────────────────────────────────────
    if args.validate:
        logger.info("-- Step 5/5: Walk-forward validation --")
        from src.training.validation import run_walk_forward_validation
        t0 = time.time()
        report = run_walk_forward_validation(
            df_raw,
            start_year=args.val_start_year,
            train_start_year=VALIDATION_TRAIN_START_YEAR,
            optimize_xgb=False,      # fast mode for validation folds
            n_optuna_trials=20,
        )
        logger.info("Validation done in %.1f min", (time.time() - t0) / 60)

        summary = report.summary()
        logger.info("Fold summary:\n%s", summary.to_string(index=False))

        means = report.mean_metrics()
        logger.info("Mean metrics: %s", means)

        # Save validation report
        out = DATA_PROCESSED_DIR / "validation_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {"folds": summary.to_dict(orient="records"), "means": means},
                f, indent=2,
            )
        logger.info("Validation report saved to %s", out)
    else:
        logger.info("-- Step 5/5: Validation SKIPPED (use --validate to enable) --")

    logger.info(
        "Training complete in %.1f min total.  Start API with:\n"
        "  uvicorn src.api.main:app --reload --port 8000",
        (time.time() - t_total) / 60,
    )


if __name__ == "__main__":
    main()
