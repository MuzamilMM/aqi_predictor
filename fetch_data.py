# ============================================================
#  fetch_data.py  —  Fetch AQI data from OpenWeather API
# ============================================================

import os, time, requests, logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from config import (
    OPENWEATHER_TOKEN, LAT, LON,
    HISTORICAL_START, HISTORICAL_CSV, DATA_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL      = "https://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL   = "https://api.openweathermap.org/data/2.5/weather"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _aqi_from_pm25(pm25: float) -> int:
    """Convert PM2.5 (µg/m³) to US AQI."""
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,   51,  100),
        (35.5,  55.4,   101, 150),
        (55.5,  150.4,  151, 200),
        (150.5, 250.4,  201, 300),
        (250.5, 350.4,  301, 400),
        (350.5, 500.4,  401, 500),
    ]
    for lo_c, hi_c, lo_i, hi_i in breakpoints:
        if lo_c <= pm25 <= hi_c:
            return round((hi_i - lo_i) / (hi_c - lo_c) * (pm25 - lo_c) + lo_i)
    return 500


def _fetch_weather(dt_unix: int) -> dict:
    """Fetch historical weather (temp, humidity, pressure, wind) for a timestamp."""
    url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
    params = {"lat": LAT, "lon": LON, "dt": dt_unix, "appid": OPENWEATHER_TOKEN, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=10)
        d = r.json()
        if "data" in d and d["data"]:
            w = d["data"][0]
            return {
                "temp":     w.get("temp"),
                "humidity": w.get("humidity"),
                "pressure": w.get("pressure"),
                "wind":     w.get("wind_speed"),
            }
    except Exception:
        pass
    return {"temp": None, "humidity": None, "pressure": None, "wind": None}


# ------------------------------------------------------------------
# Historical backfill  (chunked — API max 1 hour per call)
# ------------------------------------------------------------------

def fetch_historical(start: str = HISTORICAL_START) -> pd.DataFrame:
    """
    Pull hourly air pollution data from OpenWeather.
    The history endpoint accepts unix timestamps and returns
    up to 24 hours per call — we fetch day by day.
    """
    # Load existing data to avoid re-fetching
    existing_ts: set = set()
    if os.path.exists(HISTORICAL_CSV):
        existing_df   = pd.read_csv(HISTORICAL_CSV)
        existing_ts   = set(existing_df["datetime"].astype(str))
        log.info(f"Found {len(existing_ts)} existing records")

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.now(timezone.utc)

    records = []
    current = start_dt

    while current < end_dt:
        start_unix = int(current.timestamp())
        end_unix   = min(int(end_dt.timestamp()), start_unix + 86400)  # 24h chunk

        url    = f"{BASE_URL}/history"
        params = {"lat": LAT, "lon": LON,
                  "start": start_unix, "end": end_unix,
                  "appid": OPENWEATHER_TOKEN}

        try:
            resp = requests.get(url, params=params, timeout=15)
            body = resp.json()

            if "list" in body:
                for item in body["list"]:
                    dt_str = datetime.fromtimestamp(item["dt"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                    if dt_str in existing_ts:
                        continue

                    comp = item.get("components", {})
                    pm25 = comp.get("pm2_5", 0)

                    records.append({
                        "datetime": dt_str,
                        "aqi":      _aqi_from_pm25(pm25),
                        "pm25":     pm25,
                        "pm10":     comp.get("pm10"),
                        "co":       comp.get("co"),
                        "no2":      comp.get("no2"),
                        "o3":       comp.get("o3"),
                        "so2":      comp.get("so2"),
                        "temp":     None,
                        "humidity": None,
                        "pressure": None,
                        "wind":     None,
                    })

                date_str = current.strftime("%Y-%m-%d")
                log.info(f"  ✓ {date_str}  fetched {len(body['list'])} hourly records")
            else:
                log.warning(f"  ✗ {current.strftime('%Y-%m-%d')}  {body.get('message','no data')}")

        except Exception as e:
            log.warning(f"  ✗ {current.strftime('%Y-%m-%d')}  {e}")

        current = current + timedelta(days=1)
        time.sleep(0.3)

    # Merge with existing
    new_df = pd.DataFrame(records)
    if os.path.exists(HISTORICAL_CSV) and not new_df.empty:
        old_df = pd.read_csv(HISTORICAL_CSV)
        df = pd.concat([old_df, new_df]).drop_duplicates("datetime").sort_values("datetime")
    elif os.path.exists(HISTORICAL_CSV):
        df = pd.read_csv(HISTORICAL_CSV)
    else:
        df = new_df.sort_values("datetime") if not new_df.empty else pd.DataFrame()

    if not df.empty:
        df.to_csv(HISTORICAL_CSV, index=False)
        log.info(f"\nSaved {len(df)} total records → {HISTORICAL_CSV}")

    return df


# ------------------------------------------------------------------
# Live / real-time fetch
# ------------------------------------------------------------------

def fetch_live() -> dict:
    """Fetch current air pollution + weather for Karachi."""
    # Air pollution
    url    = f"{BASE_URL}?lat={LAT}&lon={LON}&appid={OPENWEATHER_TOKEN}"
    resp   = requests.get(url, timeout=10)
    body   = resp.json()

    if "list" not in body:
        raise RuntimeError(f"OpenWeather error: {body.get('message','unknown')}")

    item   = body["list"][0]
    comp   = item.get("components", {})
    pm25   = comp.get("pm2_5", 0)

    # Current weather
    wresp  = requests.get(WEATHER_URL, params={
        "lat": LAT, "lon": LON,
        "appid": OPENWEATHER_TOKEN, "units": "metric"
    }, timeout=10)
    wdata  = wresp.json()
    main   = wdata.get("main", {})
    wind   = wdata.get("wind", {})

    reading = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "aqi":       _aqi_from_pm25(pm25),
        "pm25":      pm25,
        "pm10":      comp.get("pm10"),
        "co":        comp.get("co"),
        "no2":       comp.get("no2"),
        "o3":        comp.get("o3"),
        "so2":       comp.get("so2"),
        "temp":      main.get("temp"),
        "humidity":  main.get("humidity"),
        "pressure":  main.get("pressure"),
        "wind":      wind.get("speed"),
        "station":   "Karachi, Pakistan",
    }

    log.info(f"Live: AQI={reading['aqi']}  PM2.5={reading['pm25']}  Temp={reading['temp']}°C")
    return reading


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Fetching historical data (Jan 2025 → today) ===")
    df = fetch_historical()

    if not df.empty:
        print(f"\n{'='*50}")
        print(f"Total records : {len(df)}")
        print(f"Date range    : {df['datetime'].min()}  →  {df['datetime'].max()}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        print(f"\nSample (last 5 rows):\n{df.tail()}")
    else:
        print("No data fetched. Check your API token.")