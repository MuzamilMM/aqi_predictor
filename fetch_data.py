import os, requests
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import certifi
from config import HISTORICAL_START, HISTORICAL_CSV, DATA_DIR, LAT, LON, MONGO_URI, MONGO_DB, RAW_COLLECTION

os.makedirs(DATA_DIR, exist_ok=True)

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def get_collection():
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=30000
    )
    client.admin.command("ping")
    return client[MONGO_DB][RAW_COLLECTION]


def fetch_historical(start=HISTORICAL_START):
    try:
        collection = get_collection()
        docs       = list(collection.find({}, {"datetime": 1}))
        existing   = {doc["datetime"] for doc in docs if "datetime" in doc}
        print(f"MongoDB connected. Existing records: {len(existing)}")
        use_mongo  = True
    except Exception as e:
        print(f"MongoDB unavailable: {e}")
        existing  = set()
        use_mongo = False
        if os.path.exists(HISTORICAL_CSV):
            existing = set(pd.read_csv(HISTORICAL_CSV)["datetime"].astype(str))
            print(f"CSV fallback: {len(existing)} existing records")

    end_date = datetime.today().strftime("%Y-%m-%d")
    params   = {
        "latitude":   LAT,
        "longitude":  LON,
        "hourly":     "us_aqi",
        "start_date": start,
        "end_date":   end_date,
    }

    print(f"Fetching AQI data from {start} to {end_date}...")
    response = requests.get(URL, params=params, timeout=30)
    data     = response.json()

    times  = data["hourly"]["time"]
    values = data["hourly"]["us_aqi"]

    new_records = [{"datetime": t, "aqi": v}
                   for t, v in zip(times, values)
                   if v is not None and t not in existing]

    if new_records:
        if use_mongo:
            try:
                collection.insert_many(new_records)
                print(f"Inserted {len(new_records)} records into MongoDB")
            except Exception as e:
                print(f"MongoDB insert failed: {e}")
        print(f"New records: {len(new_records)}")
    else:
        print("No new records")

    if use_mongo:
        try:
            all_records = list(collection.find({}, {"_id": 0}))
            df = pd.DataFrame(all_records)
        except Exception:
            df = pd.read_csv(HISTORICAL_CSV) if os.path.exists(HISTORICAL_CSV) else pd.DataFrame(new_records)
    else:
        if os.path.exists(HISTORICAL_CSV) and new_records:
            old_df = pd.read_csv(HISTORICAL_CSV)
            df     = pd.concat([old_df, pd.DataFrame(new_records)]).drop_duplicates("datetime")
        elif os.path.exists(HISTORICAL_CSV):
            df = pd.read_csv(HISTORICAL_CSV)
        else:
            df = pd.DataFrame(new_records)

    df = df.sort_values("datetime").reset_index(drop=True)
    df.to_csv(HISTORICAL_CSV, index=False)
    print(f"Total records: {len(df)}")
    return df


def fetch_live():
    params   = {"latitude": LAT, "longitude": LON, "current": "us_aqi"}
    response = requests.get(URL, params=params, timeout=10)
    data     = response.json()
    aqi      = data["current"]["us_aqi"]
    return {"aqi": aqi, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "temp": None, "humidity": None, "wind": None}


if __name__ == "__main__":
    df = fetch_historical()
    print(f"\nDate range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"\nSample:\n{df.tail()}")