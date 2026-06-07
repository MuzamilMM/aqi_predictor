# 🌫️ AQI Predictor — Karachi

**Live Dashboard:** https://aqipredictor-jwzqttnrgueww7a6a5adm3.streamlit.app/

**GitHub:** https://github.com/MuzamilMM/aqi_predictor

An end-to-end machine learning pipeline that predicts Karachi's Air Quality Index (AQI) for the next 3 days. The system fetches real data every hour, retrains models every day, and serves predictions through a publicly deployed Streamlit dashboard — fully automated via GitHub Actions with zero manual work after initial setup.

---

## 📊 Live Results (June 6, 2026)

| Date | Day | Predicted AQI | Category |
|------|-----|--------------|----------|
| 2026-06-06 | Saturday | 92.9 | Moderate |
| 2026-06-07 | Sunday | 93.8 | Moderate |
| 2026-06-08 | Monday | 94.1 | Moderate |

**Current AQI:** 89 (Moderate) | **Best Model:** Ridge Regression | **RMSE:** 0.921 | **R²:** 0.9972

---

## 🎯 Why This Project

Karachi is one of the most polluted cities in the world. AQI regularly crosses 200–300 in winters, classified as Very Unhealthy to Hazardous. Yet there is no free, reliable 3-day AQI forecast for Karachi. This project fills that gap — giving residents, schools, and healthcare providers advance warning to take preventive action.

---

## 🏗️ System Architecture

```
OpenMeteo Air Quality API  +  OpenMeteo Weather Archive API
                ↓
         fetch_data.py  ←── GitHub Actions (every hour)
                ↓
    MongoDB Atlas — raw_data collection (29,976 records)
                ↓
      feature_pipeline.py  ←── GitHub Actions (every hour)
                ↓
    MongoDB Atlas — features collection (1,235 daily rows)
                ↓
         train.py  ←── GitHub Actions (every day at midnight)
                ↓
          models/best_model.pkl
                ↓
    predict.py → dashboard.py → Streamlit Cloud (public URL)
```

---

## 🗂️ Project Structure

```
aqi_predictor/
├── config.py                        # Central settings — coordinates, MongoDB, 18 feature columns
├── fetch_data.py                    # Fetches AQI + weather from OpenMeteo, stores in MongoDB
├── feature_pipeline.py              # Engineers 18 features, saves to MongoDB feature store
├── train.py                         # Trains 4 ML models, saves best model
├── predict.py                       # Recursive 3-day forecast with weather features
├── dashboard.py                     # Streamlit dashboard (deployed on Streamlit Cloud)
├── alerts.py                        # AQI health alert system
├── shap_analysis.py                 # SHAP feature importance analysis
├── run_pipeline.py                  # Master script to run full pipeline locally
├── requirements.txt
├── .github/workflows/
│   ├── feature_pipeline.yml         # Runs every hour
│   └── training_pipeline.yml        # Runs every day at midnight
└── models/
    ├── best_model.pkl
    ├── metadata.json
    ├── model_comparison.png
    ├── shap_importance.png          # SHAP feature importance bar chart
    └── shap_summary.png             # SHAP summary dot plot
```

---

## 📦 Tech Stack

| Category | Technology |
|----------|-----------|
| AQI Data | OpenMeteo Air Quality API (free, no API key) |
| Weather Data | OpenMeteo Archive + Forecast API |
| Feature Store | MongoDB Atlas — GCP us-east-1 |
| ML Framework | Scikit-learn |
| CI/CD | GitHub Actions (hourly + daily) |
| Dashboard | Streamlit Cloud (public deployment) |
| Explainability | SHAP KernelExplainer |
| Language | Python 3.11 |

---

## 🔧 Features Used (18 Total)

| Feature | Type | Description |
|---------|------|-------------|
| aqi_lag1, lag2, lag3, lag7, lag14 | Lag | Past AQI values (1, 2, 3, 7, 14 days ago) |
| aqi_roll3, aqi_roll7 | Rolling | 3-day and 7-day rolling averages |
| aqi_change_rate | Derived | Day-over-day AQI trend (rising or falling) |
| day_of_week, month, is_weekend | Time | Calendar-based features |
| day_of_year_sin, day_of_year_cos | Seasonal | Cyclical encoding — captures annual pollution cycles |
| temperature_2m | Weather | Daily average temperature |
| relative_humidity_2m | Weather | Daily average humidity |
| wind_speed_10m | Weather | Daily average wind speed |
| precipitation | Weather | Daily total rainfall |
| surface_pressure | Weather | Daily average atmospheric pressure |

---

## 🤖 Model Performance

| Model | RMSE | MAE | R² | CV RMSE |
|-------|------|-----|----|---------|
| **Ridge Regression ✅** | **0.921** | **0.714** | **0.9972** | **43.405** |
| Lasso Regression | 1.487 | 1.230 | 0.9927 | 1.784 |
| Random Forest | 4.208 | 2.368 | 0.9415 | 8.003 |
| Gradient Boosting | 3.506 | 1.908 | 0.9594 | 7.482 |

- **Training data:** 1,235 daily rows (January 2023 — June 2026)
- **Train/Test split:** 988 training / 247 test rows (chronological, no shuffling)
- **Cross-validation:** TimeSeriesSplit with 5 folds
---

## 🔮 Forecasting Approach

Recursive multi-step prediction:
1. Day+1 predicted using real historical lags + tomorrow's weather forecast from OpenMeteo
2. Day+1 prediction becomes lag1 for Day+2
3. Day+2 prediction becomes lag1 for Day+3

Each day uses its own weather forecast — producing genuinely different predictions per day.

**Before weather features:** 93.9, 93.9, 93.9 (identical — not useful)
**After weather features:** 92.9, 93.8, 94.1 (different — reflects real atmospheric conditions)

---

## 🔍 SHAP Feature Importance

SHAP (SHapley Additive exPlanations) was used to explain what drives model predictions.

| Feature | SHAP Value | Interpretation |
|---------|-----------|----------------|
| aqi_lag1 | 25.34 | Yesterday's AQI is the strongest single predictor |
| aqi_change_rate | 13.84 | Rising vs falling trend matters more than the absolute value |
| aqi_roll3 | 8.37 | Recent 3-day average adds useful context |
| Weather features | varies | Temperature and pressure affect AQI differently by season |

**Key finding:** Day of week and is_weekend scored near zero — Karachi's AQI has no weekly pattern unlike traffic-driven cities. SHAP plots saved in `models/shap_importance.png` and `models/shap_summary.png`.

---

## ⚙️ CI/CD Pipeline (GitHub Actions)

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| Feature Pipeline | Every hour | Fetches new AQI from OpenMeteo, fetches weather, rebuilds 18 features, saves to MongoDB |
| Training Pipeline | Daily at midnight | Retrains all 4 models on latest data, saves best model as GitHub artifact |

**33+ successful workflow runs** visible in the Actions tab.

MongoDB URI stored as GitHub Actions Secret and Streamlit Cloud Secret — never written in source code.

---

## 🚧 Challenges Faced and How They Were Solved

### 1. Data Leakage (R² = 0.9997 — too perfect)
When training on hourly data, the model's lag1 feature was "1 hour ago." Since consecutive hours barely change, the model learned to just copy the previous value rather than genuinely predict. This was discovered by noticing the suspiciously perfect R² of 0.9997. The fix was switching to daily aggregation — averaging 24 hourly readings into one daily value. Now lag1 means "yesterday's average," which is a genuine 24-hour prediction challenge. R² settled to a realistic 0.9972.

### 2. Identical 3-Day Predictions
All three forecast days were showing the same value (e.g., 93.9, 93.9, 93.9). The root cause was that only the day-of-week feature changed between days — and the model had learned that weekday barely affects AQI in Karachi. The fix was adding OpenMeteo's free 3-day weather forecast as input features. Now each day has different temperature, humidity, wind, and pressure values, making each prediction reflect real expected atmospheric conditions.

### 3. MongoDB SSL Error on GitHub Actions
GitHub Actions runners could not connect to MongoDB Atlas because the cluster was in the Mumbai (ap-south-1) region while GitHub's servers run on AWS us-east-1. The SSL handshake failed silently with a TLS internal error. After trying multiple certificate fixes, the real solution was creating a new MongoDB cluster on Google Cloud us-east-1 — geographically aligned with GitHub's infrastructure. Also added 0.0.0.0/0 to MongoDB Network Access since GitHub runners use dynamic IPs.

### 4. Credentials Accidentally Committed to Code
During development, the MongoDB connection string was hardcoded in config.py with the password visible. When pushed to GitHub, the security scanner immediately blocked the push. The password was rotated on MongoDB Atlas, the credential was removed from code history, and all secrets were moved to GitHub Actions Secrets and Streamlit Cloud Secrets. config.py now reads via `os.environ.get("MONGO_URI", "")` — safe empty fallback if not set.

### 5. Data Source Selection
AQICN and OpenWeather were the first choices but both require paid plans for historical data — discovered only after registering and testing. OpenMeteo was found as a completely free alternative providing historical AQI back to 2020 with no API key required. It also provides weather archive and forecast data through separate endpoints, making it a single-source solution for all data needs.

---

## 📈 Dashboard Features

**Live at:** https://aqipredictor-jwzqttnrgueww7a6a5adm3.streamlit.app/

- Live AQI gauge with current reading
- Health alert banners — color-coded by AQI category with action advice
- 3-day forecast bar chart — different color per AQI category
- Forecast cards — day name, date, predicted AQI, category
- Model validation chart — actual vs predicted for last 60 days (MAE, RMSE, R²)
- Historical AQI chart — Karachi from January 2023 to today
- All-models comparison — side-by-side forecast from all 4 models
- Model performance table — RMSE, MAE, R², CV RMSE for all 4 models
- SHAP feature importance plots

---

## 🚨 AQI Scale

| AQI | Category | Health Action |
|-----|----------|--------------|
| 0–50 | Good | No action needed |
| 51–100 | Moderate | Sensitive people reduce outdoor activity |
| 101–150 | Unhealthy for Sensitive Groups | Sensitive groups stay indoors |
| 151–200 | Unhealthy | Everyone reduce outdoor activity |
| 201–300 | Very Unhealthy | Everyone stay indoors |
| 301–500 | Hazardous | Health emergency |

---

## 🛠️ Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set MongoDB URI (required)
$env:MONGO_URI = "your_mongodb_connection_string"

# 3. Run full pipeline
python fetch_data.py
python feature_pipeline.py
python train.py
python predict.py

# 4. Launch dashboard
streamlit run dashboard.py
```

---

## 🔭 Future Improvements

- Add XGBoost/LightGBM for potentially better performance
- Show prediction confidence intervals instead of single-point estimates
- Extend pipeline to Lahore and Islamabad
- Add email/SMS alerts when hazardous AQI is forecast
- Add EDA notebook with seasonal analysis

---

*Data: OpenMeteo · City: Karachi, Pakistan (24.8607°N, 67.0011°E) · Updated: June 2026*