import json, joblib
import numpy as np
import pandas as pd
from datetime import timedelta

from config           import MODELS_DIR, HISTORICAL_CSV, FEATURE_COLS, FORECAST_DAYS
from feature_pipeline import aqi_category, load_features_from_mongo


def load_best_model():
    return joblib.load(f"{MODELS_DIR}/best_model.pkl")


def load_all_models():
    import os, glob
    models = {}
    for path in glob.glob(f"{MODELS_DIR}/*.pkl"):
        name = os.path.basename(path).replace(".pkl", "").replace("_", " ").title()
        if name == "Best Model":
            continue
        models[name] = joblib.load(path)
    return models


def load_metadata():
    with open(f"{MODELS_DIR}/metadata.json") as f:
        return json.load(f)


def _build_feature_row(df, target_date):
    return {
        "aqi_lag1":        df["aqi"].iloc[-1],
        "aqi_lag2":        df["aqi"].iloc[-2]  if len(df) > 1  else df["aqi"].iloc[-1],
        "aqi_lag3":        df["aqi"].iloc[-3]  if len(df) > 2  else df["aqi"].iloc[-1],
        "aqi_lag7":        df["aqi"].iloc[-7]  if len(df) > 6  else df["aqi"].iloc[-1],
        "aqi_lag14":       df["aqi"].iloc[-14] if len(df) > 13 else df["aqi"].iloc[-1],
        "aqi_roll3":       df["aqi"].iloc[-3:].mean(),
        "aqi_roll7":       df["aqi"].iloc[-7:].mean(),
        "aqi_change_rate": df["aqi"].iloc[-1] - df["aqi"].iloc[-2] if len(df) > 1 else 0,
        "day_of_week":     pd.Timestamp(target_date).dayofweek,
        "month":           pd.Timestamp(target_date).month,
        "is_weekend":      int(pd.Timestamp(target_date).dayofweek >= 5),
    }


def forecast_next_days(model, days=FORECAST_DAYS):
    df        = load_features_from_mongo()
    df        = df.sort_values("datetime").reset_index(drop=True)
    last_date = df["datetime"].iloc[-1]

    forecasts = []
    for step in range(1, days + 1):
        forecast_date = pd.Timestamp(last_date) + timedelta(days=step)
        fv            = _build_feature_row(df, forecast_date)
        X             = np.array([[fv[c] for c in FEATURE_COLS]])
        aqi_pred      = max(0, float(model.predict(X)[0]))
        label, color  = aqi_category(aqi_pred)

        forecasts.append({
            "date":     forecast_date.strftime("%Y-%m-%d"),
            "day":      forecast_date.strftime("%A"),
            "aqi":      round(aqi_pred, 1),
            "category": label,
            "color":    color,
        })

    return forecasts


def forecast_all_models(days=FORECAST_DAYS):
    models = load_all_models()
    return {name: forecast_next_days(model, days)
            for name, model in models.items()}


def print_forecast(forecasts, model_name="Best Model"):
    print(f"\n{'='*55}")
    print(f"  3-DAY AQI FORECAST - Karachi  [{model_name}]")
    print(f"{'='*55}")
    for f in forecasts:
        print(f"  {f['date']}  {f['day']:<10}  AQI={f['aqi']:>6.1f}  {f['category']}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    model     = load_best_model()
    meta      = load_metadata()
    forecasts = forecast_next_days(model)
    print_forecast(forecasts, meta["best_model"])

    print("All-model comparison:")
    for mname, fc in forecast_all_models().items():
        print(f"\n  [{mname}]")
        for day in fc:
            print(f"    {day['date']}  AQI={day['aqi']}  {day['category']}")