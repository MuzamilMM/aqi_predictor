# ============================================================
#  predict.py  —  Generate 3-day AQI forecast for Karachi
# ============================================================

import json, logging, joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config           import MODELS_DIR, HISTORICAL_CSV, FEATURE_COLS, FORECAST_DAYS
from feature_pipeline import aqi_category

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Load models
# ------------------------------------------------------------------

def load_best_model():
    return joblib.load(f"{MODELS_DIR}/best_model.pkl")


def load_all_models() -> dict:
    import os, glob
    models = {}
    for path in glob.glob(f"{MODELS_DIR}/*.pkl"):
        name = os.path.basename(path).replace(".pkl", "").replace("_", " ").title()
        if name == "Best Model":
            continue
        models[name] = joblib.load(path)
    return models


def load_metadata() -> dict:
    with open(f"{MODELS_DIR}/metadata.json") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Build one feature row for a forecast step
# ------------------------------------------------------------------

def _build_step_features(df: pd.DataFrame, target_date) -> dict:
    """Build a single feature row from the rolling history DataFrame."""
    return {
        "pm10":              df["pm10"].iloc[-1],
        "co":                df["co"].iloc[-1],
        "no2":               df["no2"].iloc[-1],
        "o3":                df["o3"].iloc[-1],
        "so2":               df["so2"].iloc[-1],
        "day_of_week":       pd.Timestamp(target_date).dayofweek,
        "month":             pd.Timestamp(target_date).month,
        "is_weekend":        int(pd.Timestamp(target_date).dayofweek >= 5),
        "pm25_lag1":         df["pm25"].iloc[-1],
        "pm25_lag2":         df["pm25"].iloc[-2] if len(df) > 1 else df["pm25"].iloc[-1],
        "pm25_lag3":         df["pm25"].iloc[-3] if len(df) > 2 else df["pm25"].iloc[-1],
        "pm25_lag7":         df["pm25"].iloc[-7] if len(df) > 6 else df["pm25"].iloc[-1],
        "pm25_lag14":        df["pm25"].iloc[-14] if len(df) > 13 else df["pm25"].iloc[-1],
        "pm25_roll3":        df["pm25"].iloc[-3:].mean(),
        "pm25_roll7":        df["pm25"].iloc[-7:].mean(),
        "pm25_change_rate":  df["pm25"].iloc[-1] - df["pm25"].iloc[-2] if len(df) > 1 else 0,
    }


# ------------------------------------------------------------------
# Multi-step recursive forecast
# ------------------------------------------------------------------

def forecast_next_days(model, history_df: pd.DataFrame, days: int = FORECAST_DAYS) -> list:
    """
    Predict next `days` days recursively.
    Each predicted value feeds into the next step as a lag feature.
    """
    # Use daily averages from history
    df = history_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # If hourly data passed in, resample to daily first
    if df["datetime"].dt.hour.nunique() > 1:
        df["date"] = df["datetime"].dt.date
        df = df.groupby("date").agg(
            pm25=("pm25","mean"), pm10=("pm10","mean"),
            co=("co","mean"),    no2=("no2","mean"),
            o3=("o3","mean"),    so2=("so2","mean"),
        ).reset_index()
        df["datetime"] = pd.to_datetime(df["date"])

    df = df.sort_values("datetime").reset_index(drop=True)
    last_date = df["datetime"].iloc[-1]

    forecasts = []

    for step in range(1, days + 1):
        forecast_date = last_date + timedelta(days=step)
        fv = _build_step_features(df, forecast_date)
        X  = np.array([[fv[c] for c in FEATURE_COLS]])

        pm25_pred = float(model.predict(X)[0])
        pm25_pred = max(0, pm25_pred)

        label, color = aqi_category(pm25_pred)

        forecasts.append({
            "date":     forecast_date.strftime("%Y-%m-%d"),
            "day":      forecast_date.strftime("%A"),
            "pm25":     round(pm25_pred, 1),
            "aqi":      round(pm25_pred),
            "category": label,
            "color":    color,
        })

        # Append predicted row so next step can use it as lag
        new_row = pd.DataFrame([{
            "datetime": forecast_date,
            "pm25":     pm25_pred,
            "pm10":     df["pm10"].iloc[-1],
            "co":       df["co"].iloc[-1],
            "no2":      df["no2"].iloc[-1],
            "o3":       df["o3"].iloc[-1],
            "so2":      df["so2"].iloc[-1],
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    return forecasts


def forecast_all_models(history_df: pd.DataFrame, days: int = FORECAST_DAYS) -> dict:
    models = load_all_models()
    return {name: forecast_next_days(model, history_df, days)
            for name, model in models.items()}


# ------------------------------------------------------------------
# Pretty print
# ------------------------------------------------------------------

def print_forecast(forecasts: list, model_name: str = "Best Model"):
    print(f"\n{'='*55}")
    print(f"  3-DAY AQI FORECAST — Karachi  [{model_name}]")
    print(f"{'='*55}")
    for f in forecasts:
        bar = "█" * min(int(f["pm25"] / 10), 30)
        print(f"  {f['date']}  {f['day']:<10}  PM2.5={f['pm25']:>6.1f}  {bar}")
        print(f"    Category: {f['category']}")
    print(f"{'='*55}\n")


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    history  = pd.read_csv(HISTORICAL_CSV, parse_dates=["datetime"])
    model    = load_best_model()
    meta     = load_metadata()

    forecasts = forecast_next_days(model, history)
    print_forecast(forecasts, meta["best_model"])

    print("All-model comparison:")
    all_f = forecast_all_models(history)
    for mname, fc in all_f.items():
        print(f"\n  [{mname}]")
        for day in fc:
            print(f"    {day['date']}  PM2.5={day['pm25']}  {day['category']}")