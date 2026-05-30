import os, requests
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from config import HISTORICAL_START, HISTORICAL_CSV, DATA_DIR, LAT, LON, MONGO_URI, MONGO_DB, RAW_COLLECTION

os.makedirs(DATA_DIR, exist_ok=True)

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def get_collection():
    client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
    return client[MONGO_DB][RAW_COLLECTION]


def fetch_historical(start=HISTORICAL_START):
    collection = get_collection()

    existing = set(doc["datetime"] for doc in collection.find({}, {"datetime": 1}))
    print(f"Existing records in MongoDB: {len(existing)}")

    end_date = datetime.today().strftime("%Y-%m-%d")

    params = {
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

    new_records = []
    for t, v in zip(times, values):
        if v is not None and t not in existing:
            new_records.append({"datetime": t, "aqi": v})

    if new_records:
        collection.insert_many(new_records)
        print(f"Inserted {len(new_records)} new records into MongoDB")
    else:
        print("No new records to insert")

    all_records = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(all_records).sort_values("datetime").reset_index(drop=True)
    df.to_csv(HISTORICAL_CSV, index=False)
    print(f"Total records: {len(df)}")
    return df


def fetch_live():
    params = {
        "latitude":  LAT,
        "longitude": LON,
        "current":   "us_aqi",
    }
    response = requests.get(URL, params=params, timeout=10)
    data     = response.json()
    aqi      = data["current"]["us_aqi"]
    print(f"Live AQI: {aqi}")
    return {"aqi": aqi, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "temp": None, "humidity": None, "wind": None}


if __name__ == "__main__":
    df = fetch_historical()
    print(f"\nDate range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"\nSample:\n{df.tail()}")