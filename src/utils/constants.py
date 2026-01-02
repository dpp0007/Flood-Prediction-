"""
Constants and configuration for Flood Early Warning System.

Data Source: Central Water Commission (CWC), Government of India
Website: https://inf.cwc.gov.in/
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# CWC Data Source Attribution
CWC_BASE_URL = "https://inf.cwc.gov.in"
CWC_ATTRIBUTION = {
    "source": "Central Water Commission (CWC), Government of India",
    "url": "https://inf.cwc.gov.in/",
    "disclaimer": "This system does NOT replace official CWC flood warnings. "
                  "Intended for academic, research, and demonstration use only.",
}

# Scraping configuration
SCRAPE_INTERVAL_MINUTES = 30
REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2
USER_AGENT = "Flood-Warning-System/1.0 (Research Purpose)"

# Data validation thresholds
MAX_WATER_LEVEL_JUMP_METERS = 2.0  # Detect sensor errors
MIN_VALID_LEVEL = -10.0
MAX_VALID_LEVEL = 100.0

# Feature engineering
LOOKBACK_HOURS = 24
RATE_OF_RISE_WINDOW_HOURS = 3
HISTORICAL_PERCENTILE_WINDOW_DAYS = 365

# Risk tier thresholds (0-1 scale)
RISK_TIERS = {
    "low": (0.0, 0.33),
    "medium": (0.33, 0.67),
    "high": (0.67, 1.0),
}

# Model configuration
TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.1
RANDOM_STATE = 42

# Warning prediction horizon
WARNING_THRESHOLD_HOURS = 6

# Feature importance
TOP_FEATURES_FOR_EXPLANATION = 5

# Feature names for ML models
FEATURE_NAMES = [
    "current_level",
    "distance_to_warning",
    "distance_to_danger",
    "rate_of_rise_3h",
    "consecutive_rising_hours",
    "historical_percentile",
    "basin_avg_level",
    "station_volatility",
    "hours_since_last_update",
    "seasonal_factor",
]

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
