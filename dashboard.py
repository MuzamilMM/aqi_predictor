# ============================================================
#  dashboard.py  —  Streamlit AQI Dashboard for Karachi
#  Run: streamlit run dashboard.py
# ============================================================

import json, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from config           import MODELS_DIR, HISTORICAL_CSV, FEATURES_CSV, CITY_NAME, FEATURE_COLS, TARGET_COL
from fetch_data       import fetch_live
from feature_pipeline import aqi_category
from predict          import load_best_model, load_all_models, load_metadata, forecast_next_days, forecast_all_models


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title=f"AQI Predictor — {CITY_NAME}",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.forecast-card {
    background: #2a2a3e; border-radius: 10px; padding: 16px;
    margin: 6px; text-align: center;
}
.aqi-value { font-size: 3rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_history():
    if not os.path.exists(HISTORICAL_CSV):
        st.error("No historical data found. Run fetch_data.py first.")
        st.stop()
    df = pd.read_csv(HISTORICAL_CSV, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_features():
    if not os.path.exists(FEATURES_CSV):
        st.error("No feature data found. Run feature_pipeline.py first.")
        st.stop()
    df = pd.read_csv(FEATURES_CSV, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_resource
def load_model_meta():
    if not os.path.exists(f"{MODELS_DIR}/metadata.json"):
        st.error("No trained models found. Run train.py first.")
        st.stop()
    meta   = load_metadata()
    model  = load_best_model()
    models = load_all_models()
    return model, models, meta


@st.cache_data(ttl=600)
def get_live_reading():
    try:
        return fetch_live()
    except Exception as e:
        st.warning(f"Live fetch failed: {e}. Using latest historical data.")
        return None


@st.cache_data(ttl=3600)
def get_shap_data(model_name: str):
    """Compute SHAP values (cached so it only runs once)."""
    try:
        import shap, joblib
        df       = load_features().dropna(subset=FEATURE_COLS + [TARGET_COL])
        X        = df[FEATURE_COLS]
        pipeline = joblib.load(f"{MODELS_DIR}/{model_name}.pkl")
        scaler   = pipeline.named_steps["scaler"]
        model    = pipeline.named_steps["model"]
        X_scaled = scaler.transform(X)
        sample   = shap.sample(X_scaled, 150, random_state=42)
        explainer   = shap.KernelExplainer(model.predict, sample)
        shap_values = explainer.shap_values(sample, nsamples=80)
        mean_shap   = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Importance": mean_shap,
        }).sort_values("Importance", ascending=True)
    except Exception as e:
        return None


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

def sidebar():
    st.sidebar.header("⚙️ Settings")
    model_choice    = st.sidebar.selectbox(
        "Forecast model",
        ["Best (auto)", "Ridge Regression", "Lasso Regression",
         "Random Forest", "Gradient Boosting"],
    )
    show_all_models = st.sidebar.checkbox("Compare all models",    value=False)
    show_validation = st.sidebar.checkbox("Show model validation", value=True)
    show_shap       = st.sidebar.checkbox("Show SHAP analysis",    value=False)
    refresh         = st.sidebar.button("🔄 Refresh live data")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**City:** {CITY_NAME}")
    st.sidebar.markdown(f"**Source:** OpenWeather API")
    st.sidebar.markdown(f"**Note:** Predictions may differ from AQICN by 10-20 units due to different data sources (satellite vs ground sensor).")
    st.sidebar.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    return model_choice, show_all_models, show_validation, show_shap, refresh


# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------

def aqi_gauge(value: float) -> go.Figure:
    label, color = aqi_category(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": f"Current AQI<br><span style='font-size:0.8em;color:{color}'>{label}</span>",
               "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 500]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,   50],  "color": "#00e400"},
                {"range": [50,  100], "color": "#ffff00"},
                {"range": [100, 150], "color": "#ff7e00"},
                {"range": [150, 200], "color": "#ff0000"},
                {"range": [200, 300], "color": "#8f3f97"},
                {"range": [300, 500], "color": "#7e0023"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    return fig


def historical_chart(df: pd.DataFrame) -> go.Figure:
    daily = df.copy()
    daily["date"] = daily["datetime"].dt.date
    daily = daily.groupby("date")["pm25"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["pm25"],
        mode="lines", name="PM2.5 (daily avg)",
        line=dict(color="#4fc3f7", width=1.5),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.1)",
    ))
    for val, color, label in [(50,"#00e400","Good"),(100,"#ffff00","Moderate"),
                               (150,"#ff7e00","USG"),(200,"#ff0000","Unhealthy")]:
        fig.add_hline(y=val, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="right",
                      line_width=0.8, opacity=0.6)
    fig.update_layout(
        title="Historical PM2.5 — Karachi (Daily Average)",
        xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)",
        height=360, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


def forecast_chart(forecasts: list, title: str) -> go.Figure:
    dates  = [f["date"] for f in forecasts]
    values = [f["pm25"] for f in forecasts]
    colors = [f["color"] for f in forecasts]
    labels = [f"{v}<br>{forecasts[i]['category']}" for i, v in enumerate(values)]
    fig = go.Figure(go.Bar(
        x=dates, y=values, marker_color=colors,
        text=labels, textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)",
        height=320, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="white",
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


def model_comparison_chart(all_forecasts: dict) -> go.Figure:
    colors = {"Lasso Regression":"#4fc3f7", "Ridge Regression":"#81c784",
              "Random Forest":"#ffb74d",    "Gradient Boosting":"#f06292"}
    fig = go.Figure()
    for mname, forecasts in all_forecasts.items():
        fig.add_trace(go.Scatter(
            x=[f["date"] for f in forecasts],
            y=[f["pm25"] for f in forecasts],
            mode="lines+markers", name=mname,
            line=dict(color=colors.get(mname, "#fff"), width=2),
            marker=dict(size=8),
        ))
    fig.update_layout(
        title="All Models — 3-Day PM2.5 Forecast Comparison",
        xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)",
        height=360, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


def validation_chart(features_df: pd.DataFrame, model) -> tuple:
    df = features_df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    df = df.sort_values("datetime").tail(60)
    df["predicted"] = model.predict(df[FEATURE_COLS]).clip(min=0)

    mae  = round((df[TARGET_COL] - df["predicted"]).abs().mean(), 2)
    rmse = round(((df[TARGET_COL] - df["predicted"]) ** 2).mean() ** 0.5, 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df[TARGET_COL],
        mode="lines", name="Actual PM2.5",
        line=dict(color="#4fc3f7", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["predicted"],
        mode="lines", name="Predicted PM2.5",
        line=dict(color="#FF5722", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=f"Model Validation — Last 60 Days  (MAE={mae} µg/m³  |  RMSE={rmse} µg/m³)",
        xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)",
        height=380, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig, mae, rmse


def shap_chart(shap_df: pd.DataFrame, model_name: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=shap_df["Importance"],
        y=shap_df["Feature"],
        orientation="h",
        marker_color="#4fc3f7",
        text=shap_df["Importance"].round(3),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"SHAP Feature Importance — {model_name}",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    return fig


# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------

def main():
    model_choice, show_all_models, show_validation, show_shap, refresh = sidebar()

    st.title(f"🌫️ AQI Forecaster — {CITY_NAME}")
    st.markdown("**Real-time air quality monitoring and 3-day PM2.5 prediction**")
    st.markdown("---")

    history     = load_history()
    features_df = load_features()
    best_model, all_models, meta = load_model_meta()

    if refresh:
        st.cache_data.clear()

    live = get_live_reading()

    # Select model
    if model_choice == "Best (auto)":
        selected_model = best_model
        selected_name  = meta["best_model"]
    else:
        selected_model = all_models.get(model_choice, best_model)
        selected_name  = model_choice

    forecasts     = forecast_next_days(selected_model, history)
    all_forecasts = forecast_all_models(history) if show_all_models else {}

    # ── KPIs ────────────────────────────────────────────────────
    latest      = history.sort_values("datetime").iloc[-1]
    current_pm  = live["pm25"] if live else latest["pm25"]
    current_aqi = live["aqi"]  if live else latest["aqi"]
    label, _    = aqi_category(current_aqi)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌫️ AQI Now",     round(current_aqi), label)
    c2.metric("💨 PM2.5",       f"{round(current_pm, 1)} µg/m³")
    c3.metric("🌡️ Temperature",  f"{live['temp']}°C"   if live else "—")
    c4.metric("💧 Humidity",    f"{live['humidity']}%" if live else "—")
    c5.metric("🌬️ Wind",        f"{live['wind']} m/s"  if live else "—")

    st.markdown("---")

    # ── Gauge + Forecast ─────────────────────────────────────────
    col_g, col_f = st.columns([1, 2])
    with col_g:
        st.plotly_chart(aqi_gauge(current_aqi), use_container_width=True)
    with col_f:
        st.plotly_chart(forecast_chart(forecasts, f"3-Day Forecast [{selected_name}]"),
                        use_container_width=True)

    # ── Forecast cards ───────────────────────────────────────────
    st.subheader("📅 Daily Forecast Detail")
    cols = st.columns(3)
    for i, fc in enumerate(forecasts):
        _, color = aqi_category(fc["pm25"])
        with cols[i]:
            st.markdown(f"""
<div class="forecast-card" style="border-top: 4px solid {color}">
  <div style="font-size:1.1rem;font-weight:600">{fc['day']}</div>
  <div style="color:#aaa;font-size:0.85rem">{fc['date']}</div>
  <div class="aqi-value" style="color:{color}">{fc['pm25']}</div>
  <div style="font-size:0.8rem;color:#ccc">µg/m³ PM2.5</div>
  <div style="margin-top:8px;font-size:0.85rem">{fc['category']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Validation ───────────────────────────────────────────────
    if show_validation:
        st.subheader("✅ Model Validation — Actual vs Predicted")
        st.caption("How accurately the model predicted PM2.5 on past data it has already seen (last 60 days)")
        fig_val, mae, rmse = validation_chart(features_df, selected_model)
        st.plotly_chart(fig_val, use_container_width=True)

        v1, v2, v3 = st.columns(3)
        v1.metric("📉 MAE",  f"{mae} µg/m³",  help="Average prediction error")
        v2.metric("📉 RMSE", f"{rmse} µg/m³", help="Penalizes large errors more")
        v3.metric("📈 R²",   f"{meta['summary'][selected_name]['R2']}",
                  help="1.0 = perfect prediction")

        st.info("💡 Note: Live AQI from sites like AQICN may differ by 10-20 units — this is normal because they use physical ground sensors while our model is trained on OpenWeather satellite data.")
        st.markdown("---")

    # ── SHAP ─────────────────────────────────────────────────────
    if show_shap:
        st.subheader("🔍 SHAP Feature Importance")
        st.caption("Which features influence the model's predictions the most")

        safe_name = selected_name.lower().replace(" ", "_")
        with st.spinner("Computing SHAP values... (this takes ~1 minute)"):
            shap_df = get_shap_data(safe_name)

        if shap_df is not None:
            st.plotly_chart(shap_chart(shap_df, selected_name), use_container_width=True)
            st.caption("Higher SHAP value = stronger influence on prediction. Positive = pushes AQI up, Negative = pushes AQI down.")
        else:
            st.warning("SHAP computation failed. Try a different model.")

        st.markdown("---")

    # ── Historical ───────────────────────────────────────────────
    st.plotly_chart(historical_chart(history), use_container_width=True)

    # ── Model comparison ─────────────────────────────────────────
    if show_all_models and all_forecasts:
        st.plotly_chart(model_comparison_chart(all_forecasts), use_container_width=True)

    # ── Metrics table ────────────────────────────────────────────
    st.subheader("📊 Model Performance")
    rows = []
    for mname, m in meta["summary"].items():
        rows.append({
            "Model":   mname,
            "RMSE":    m["RMSE"],
            "MAE":     m["MAE"],
            "R²":      m["R2"],
            "CV RMSE": m["CV_RMSE"],
            "Best":    "✅" if mname == meta["best_model"] else "",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#666;font-size:0.8rem'>"
        f"Data: OpenWeather API · Karachi, Pakistan · {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        f"</div>", unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()