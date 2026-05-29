# ============================================================
#  config.py  —  Central configuration for AQI Predictor
# ============================================================

# --- OpenWeather API ---
# Token is stored in GitHub Secrets (OPENWEATHER_TOKEN)
# For local use, create a .env file or set the environment variable
import os
OPENWEATHER_TOKEN = os.environ.get("OPENWEATHER_TOKEN", "")

# --- Karachi coordinates ---
CITY_NAME = "Karachi"
LAT       = 24.8607
LON       = 67.0011

# --- Data settings ---
HISTORICAL_START = "2025-01-01"
DATA_DIR         = "data"
HISTORICAL_CSV   = f"{DATA_DIR}/karachi_historical.csv"
FEATURES_CSV     = f"{DATA_DIR}/karachi_features.csv"

# --- Feature columns (daily data — no hour feature) ---
FEATURE_COLS = [
    # Pollutants
    "pm10", "co", "no2", "o3", "so2",
    # Time
    "day_of_week", "month", "is_weekend",
    # Lag features (now = days ago, not hours)
    "pm25_lag1", "pm25_lag2", "pm25_lag3",
    "pm25_lag7", "pm25_lag14",
    # Rolling averages
    "pm25_roll3", "pm25_roll7",
    # Rate of change
    "pm25_change_rate",
]

TARGET_COL = "pm25"

# --- Model settings ---
MODELS_DIR    = "models"
FORECAST_DAYS = 3
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# --- Dashboard ---
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501