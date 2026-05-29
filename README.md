# AQI Predictor — Karachi
End-to-end ML pipeline for 3-day PM2.5 / AQI forecasting.

## Project Structure
```
aqi_predictor/
├── config.py              # All settings (token, paths, features)
├── fetch_data.py          # Pull historical + live data from AQICN
├── feature_pipeline.py    # Feature engineering (lags, rolling, time)
├── train.py               # Train 4 models, save best
├── predict.py             # 3-day recursive forecast
├── dashboard.py           # Streamlit web dashboard
├── run_pipeline.py        # Run everything end-to-end
├── requirements.txt
├── data/                  # CSVs saved here
└── models/                # Trained .pkl files saved here
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Add your token
Edit `config.py` line 7:
```python
AQICN_TOKEN = "your_actual_token_here"
```
Or set the token as an environment variable:
```bash
set AQICN_TOKEN=your_actual_token_here
```
If you leave the placeholder `your_token_here`, `fetch_data.py` will raise an error and tell you to configure the token.

## Step 2 — Run full pipeline
```bash
python run_pipeline.py
```
This will:
1. Fetch ~2 years of daily Karachi AQI data (~700 API calls, takes ~5 min)
2. Engineer features (lags, rolling averages, time features)
3. Train 4 models: Ridge, Lasso, Random Forest, Gradient Boosting
4. Print a 3-day forecast table
5. Tell you how to launch the dashboard

## Step 3 — Launch dashboard
```bash
streamlit run dashboard.py
```
Opens at http://localhost:8501

## Re-run options
```bash
# Skip fetching (data already downloaded):
python run_pipeline.py --skip-fetch

# Skip training (models already saved):
python run_pipeline.py --skip-train --skip-fetch
```

## Models
| Model              | Description                              |
|--------------------|------------------------------------------|
| Ridge Regression   | Linear with L2 regularization            |
| Lasso Regression   | Linear with L1 (sparse features)         |
| Random Forest      | 200 trees, handles nonlinearity well     |
| Gradient Boosting  | Sequential boosting, usually most accurate|

## Features Used
- Weather: temperature, humidity, pressure, wind speed, dew point
- Time: hour, day of week, month, is_weekend
- Lag: PM2.5 from 1, 2, 3, 7, 14 days ago
- Rolling: 3-day and 7-day PM2.5 average
- Derived: PM2.5 change rate, temp×humidity interaction
