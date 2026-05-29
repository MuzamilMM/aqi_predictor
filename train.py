# ============================================================
#  train.py  —  Train & evaluate 4 ML models, save best
# ============================================================

import os, json, logging, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection       import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing         import StandardScaler
from sklearn.linear_model          import Ridge, Lasso
from sklearn.ensemble              import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics               import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline              import Pipeline

from config import (
    FEATURES_CSV, MODELS_DIR, FEATURE_COLS, TARGET_COL,
    TEST_SIZE, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Model definitions
# ------------------------------------------------------------------

def get_models() -> dict:
    return {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=1.0, max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ------------------------------------------------------------------
# Metrics helper
# ------------------------------------------------------------------

def evaluate(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 4)}


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_all() -> dict:
    # 1. Load features
    df = pd.read_csv(FEATURES_CSV, parse_dates=["datetime"])
    df = df.sort_values("datetime").dropna(subset=FEATURE_COLS + [TARGET_COL])
    log.info(f"Loaded {len(df)} rows for training")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 2. Time-aware train/test split
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    log.info(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # 3. Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    models    = get_models()
    results   = {}
    best_name = None
    best_rmse = float("inf")

    for name, pipeline in models.items():
        log.info(f"Training: {name} ...")

        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        cv_rmse = -cv_scores.mean()

        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics["CV_RMSE"] = round(cv_rmse, 3)

        results[name] = {
            "metrics":  metrics,
            "y_test":   y_test.tolist(),
            "y_pred":   y_pred.tolist(),
            "pipeline": pipeline,
        }

        log.info(f"  → RMSE={metrics['RMSE']}  MAE={metrics['MAE']}  R²={metrics['R2']}  CV_RMSE={metrics['CV_RMSE']}")

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_name = name

    log.info(f"\n✓ Best model: {best_name}  (RMSE={best_rmse})")

    # 4. Save all models + metadata
    summary = {}
    for name, result in results.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(result["pipeline"], f"{MODELS_DIR}/{safe}.pkl")
        summary[name] = result["metrics"]

    best_pipeline = results[best_name]["pipeline"]
    joblib.dump(best_pipeline, f"{MODELS_DIR}/best_model.pkl")

    meta = {
        "best_model":   best_name,
        "best_rmse":    best_rmse,
        "feature_cols": FEATURE_COLS,
        "target":       TARGET_COL,
        "train_rows":   len(X_train),
        "test_rows":    len(X_test),
        "summary":      summary,
    }
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Models saved to '{MODELS_DIR}/'")

    # 5. Plot results
    _plot_results(results, df["datetime"].iloc[split_idx:].values)

    # 6. Feature importance
    _print_feature_importance(results)

    return meta


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def _plot_results(results: dict, dates):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("AQI Prediction — Karachi (Actual vs Predicted)", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, (name, result) in zip(axes, results.items()):
        y_test = result["y_test"]
        y_pred = result["y_pred"]
        m      = result["metrics"]

        ax.plot(dates, y_test, label="Actual",    color="#2196F3", linewidth=1.5)
        ax.plot(dates, y_pred, label="Predicted", color="#FF5722", linewidth=1.5, linestyle="--")
        ax.set_title(f"{name}\nRMSE={m['RMSE']}  MAE={m['MAE']}  R²={m['R2']}", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("PM2.5")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"{MODELS_DIR}/model_comparison.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved → {path}")


def _print_feature_importance(results: dict):
    print(f"\n{'='*55}")
    print("FEATURE IMPORTANCE / COEFFICIENTS")
    print(f"{'='*55}")
    for name, result in results.items():
        pipeline = result["pipeline"]
        model    = pipeline.named_steps["model"]
        print(f"\n── {name} ──")
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
            print(imp.sort_values(ascending=False).head(10).round(4).to_string())
        elif hasattr(model, "coef_"):
            coef = pd.Series(model.coef_, index=FEATURE_COLS)
            print(coef.abs().sort_values(ascending=False).head(10).round(4).to_string())


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

def print_summary(meta: dict):
    print(f"\n{'='*55}")
    print(f"  TRAINING SUMMARY — {meta['train_rows']} train / {meta['test_rows']} test rows")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'CV_RMSE':>9}")
    print(f"  {'-'*55}")
    for name, m in meta["summary"].items():
        star = " ★" if name == meta["best_model"] else ""
        print(f"  {name:<25} {m['RMSE']:>7} {m['MAE']:>7} {m['R2']:>7} {m['CV_RMSE']:>9}{star}")
    print(f"\n  Best: {meta['best_model']}  (RMSE={meta['best_rmse']})")
    print(f"{'='*55}\n")


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    meta = train_all()
    print_summary(meta)