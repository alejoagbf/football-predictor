"""Central configuration for the football prediction system."""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
BAYESIAN_MODEL_DIR = MODELS_DIR / "bayesian"
XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
RESULTS_FILE = DATA_RAW_DIR / "results.csv"
FEATURES_FILE = DATA_PROCESSED_DIR / "features.parquet"

# ── ELO ─────────────────────────────────────────────────────────────────────
ELO_INITIAL_RATING: float = 1500.0
ELO_HOME_ADVANTAGE: float = 100.0  # Added to home rating in expected-score calc

# K-factors keyed by substring of tournament name (checked in order)
ELO_K_FACTORS: dict[str, float] = {
    "FIFA World Cup": 60.0,
    "Copa America": 50.0,
    "UEFA Euro": 50.0,
    "Africa Cup of Nations": 50.0,
    "Asian Cup": 50.0,
    "Gold Cup": 50.0,
    "CONCACAF": 50.0,
    "Confederations Cup": 50.0,
    "Nations League": 45.0,
    "Olympic Games": 40.0,
    "Qualification": 40.0,
    "Friendly": 30.0,
}
ELO_K_DEFAULT: float = 40.0

# ── Feature engineering ──────────────────────────────────────────────────────
FORM_WINDOWS: list[int] = [5, 10]
H2H_WINDOW: int = 10          # Last N head-to-head matches
ATTACK_DEFENSE_WINDOW: int = 20  # Rolling window for strength estimates

TOURNAMENT_CATEGORIES: dict[str, str] = {
    "FIFA World Cup": "world_cup",
    "UEFA Euro": "continental",
    "Copa America": "continental",
    "Africa Cup of Nations": "continental",
    "Asian Cup": "continental",
    "Gold Cup": "continental",
    "CONCACAF": "continental",
    "Confederations Cup": "continental",
    "Nations League": "nations_league",
    "Olympic Games": "olympics",
    "Qualification": "qualification",
    "Friendly": "friendly",
}
TOURNAMENT_CATEGORY_DEFAULT = "other"

# ── Bayesian model ───────────────────────────────────────────────────────────
BAYESIAN_DATA_YEARS: int = 10      # Only use last N years for Bayesian training
BAYESIAN_SAMPLES: int = 1000
BAYESIAN_TUNE: int = 1000
BAYESIAN_TARGET_ACCEPT: float = 0.9
BAYESIAN_CHAINS: int = 2
BAYESIAN_RANDOM_SEED: int = 42

# ── XGBoost model ────────────────────────────────────────────────────────────
XGBOOST_OPTUNA_TRIALS: int = 80
XGBOOST_RANDOM_SEED: int = 42
XGBOOST_TEMPORAL_DECAY: float = 0.0003  # Weight = exp(-decay * days_ago)

# ── Ensemble ─────────────────────────────────────────────────────────────────
ENSEMBLE_WEIGHT_BAYESIAN: float = 0.6
ENSEMBLE_WEIGHT_XGBOOST: float = 0.4

# ── Poisson matrix ───────────────────────────────────────────────────────────
POISSON_MAX_GOALS: int = 8  # Matrix covers 0..POISSON_MAX_GOALS for each team

# ── Validation ───────────────────────────────────────────────────────────────
VALIDATION_START_YEAR: int = 2016
VALIDATION_TRAIN_START_YEAR: int = 1990  # Skip very old data for validation

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
