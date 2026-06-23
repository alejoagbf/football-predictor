"""Download and load the international results dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_RAW_DIR, RESULTS_FILE, RESULTS_URL

logger = logging.getLogger(__name__)


def download_dataset(force: bool = False) -> Path:
    """Download results.csv from GitHub if not already present."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_FILE.exists() and not force:
        logger.info("Dataset already present at %s — skipping download.", RESULTS_FILE)
        return RESULTS_FILE

    logger.info("Downloading dataset from %s …", RESULTS_URL)
    response = requests.get(RESULTS_URL, timeout=60)
    response.raise_for_status()

    RESULTS_FILE.write_bytes(response.content)
    logger.info("Saved %d bytes to %s", len(response.content), RESULTS_FILE)
    return RESULTS_FILE


def load_results(path: Path | None = None) -> pd.DataFrame:
    """
    Load and clean the international results CSV.

    Returns a DataFrame with typed columns, sorted by date ascending.
    The 'neutral' column is converted to bool; dates to datetime.
    """
    csv_path = path or RESULTS_FILE

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results CSV not found at {csv_path}. Run download_dataset() first."
        )

    df = pd.read_csv(
        csv_path,
        parse_dates=["date"],
        dtype={
            "home_team": str,
            "away_team": str,
            "home_score": "Int64",
            "away_score": "Int64",
            "tournament": str,
            "city": str,
            "country": str,
        },
    )

    # Normalise neutral column regardless of source format
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(str).str.strip().str.upper() == "TRUE"
    else:
        df["neutral"] = False

    # Drop rows with missing scores (shootout or suspended matches)
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing scores.", dropped)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Derive result column: 1=home win, 0=draw, -1=away win
    df["result"] = (df["home_score"] > df["away_score"]).astype(int) - (
        df["home_score"] < df["away_score"]
    ).astype(int)

    df = df.sort_values("date").reset_index(drop=True)
    logger.info(
        "Loaded %d matches (%s – %s)",
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df
