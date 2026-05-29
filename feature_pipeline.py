# ============================================================
#  feature_pipeline.py  —  Engineer features from raw CSV
# ============================================================

import logging
import pandas as pd
import numpy as np
from config import HISTORICAL_CSV, FEATURES_CSV, FEATURE_COLS, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core engineering
# ------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : raw hourly DataFrame
    Output: daily-aggregated, feature-engineered DataFrame
    """
    df = df.copy()

    # --- Parse datetime ---
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # --- Resample to DAILY averages ---
    # This prevents data leakage from hourly lag features
    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date").agg(
        pm25     = ("pm25",     "mean"),
        pm10     = ("pm10",     "mean"),
        co       = ("co",       "mean"),
        no2      = ("no2",      "mean"),
        o3       = ("o3",       "mean"),
        so2      = ("so2",      "mean"),
        aqi      = ("aqi",      "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    log.info(f"Resampled {len(df)} hourly rows → {len(daily)} daily rows")

    # --- Forward-fill small gaps (≤3 days) ---
    numeric_cols = ["pm25","pm10","co","no2","o3","so2","aqi"]
    daily[numeric_cols] = daily[numeric_cols].ffill(limit=3)

    # --- Time features ---
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"]       = daily["date"].dt.month
    daily["is_weekend"]  = (daily["day_of_week"] >= 5).astype(int)

    # --- Lag features (now meaningful — 1 lag = 1 day ago) ---
    daily["pm25_lag1"]  = daily["pm25"].shift(1)
    daily["pm25_lag2"]  = daily["pm25"].shift(2)
    daily["pm25_lag3"]  = daily["pm25"].shift(3)
    daily["pm25_lag7"]  = daily["pm25"].shift(7)
    daily["pm25_lag14"] = daily["pm25"].shift(14)

    # --- Rolling averages ---
    daily["pm25_roll3"] = daily["pm25"].shift(1).rolling(3).mean()
    daily["pm25_roll7"] = daily["pm25"].shift(1).rolling(7).mean()

    # --- Rate of change ---
    daily["pm25_change_rate"] = daily["pm25"].diff()

    # --- Rename date to datetime for consistency ---
    daily = daily.rename(columns={"date": "datetime"})

    # --- Drop rows with NaN ---
    required = FEATURE_COLS + [TARGET_COL, "datetime"]
    daily = daily.dropna(subset=[c for c in required if c in daily.columns])
    daily = daily.reset_index(drop=True)

    log.info(f"Feature engineering done: {len(daily)} usable daily rows, {len(FEATURE_COLS)} features")
    return daily


# ------------------------------------------------------------------
# AQI category helper
# ------------------------------------------------------------------

def aqi_category(value: float) -> tuple:
    if value <= 50:
        return "Good", "#00e400"
    elif value <= 100:
        return "Moderate", "#ffff00"
    elif value <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif value <= 200:
        return "Unhealthy", "#ff0000"
    elif value <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    raw = pd.read_csv(HISTORICAL_CSV)
    log.info(f"Loaded {len(raw)} raw records from {HISTORICAL_CSV}")

    features_df = build_features(raw)
    features_df.to_csv(FEATURES_CSV, index=False)
    log.info(f"Saved feature dataset → {FEATURES_CSV}")

    print(f"\n{'='*50}")
    print(f"Shape: {features_df.shape}")
    print(f"\nFeature stats:\n{features_df[FEATURE_COLS].describe().round(2)}")
    print(f"\nCorrelation with PM2.5 (top 10):")
    corr = features_df[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    print(corr.abs().sort_values(ascending=False).head(10).round(3))