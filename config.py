import os

CITY_NAME = "Karachi"
LAT       = 24.8607
LON       = 67.0011

HISTORICAL_START = "2025-01-01"
DATA_DIR         = "data"
HISTORICAL_CSV   = f"{DATA_DIR}/karachi_historical.csv"

MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://msajjad29350_db_user:8HWuMjdWYqMp8nb7@cluster0.b44pod8.mongodb.net/?appName=Cluster0")
MONGO_DB  = "aqi_predictor"
RAW_COLLECTION      = "raw_data"
FEATURES_COLLECTION = "features"

FEATURE_COLS = [
    "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "aqi_lag7", "aqi_lag14",
    "aqi_roll3", "aqi_roll7",
    "aqi_change_rate",
    "day_of_week", "month", "is_weekend",
]

TARGET_COL    = "aqi"
MODELS_DIR    = "models"
FORECAST_DAYS = 3
TEST_SIZE     = 0.2
RANDOM_STATE  = 42