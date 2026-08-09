"""
forecasting_model.py
---------------------
Trains REAL machine learning models (scikit-learn GradientBoostingRegressor)
to forecast, for every house, the next hour's:
    - solar generation (kWh)
    - demand (kWh)

Features used (all things the digital twin would know ahead of time):
    hour, day-of-week, cloud_cover forecast, temp forecast,
    house static attributes (panel_kw, household_size, profile one-hot),
    lag features (generation/demand 1h and 24h ago)

We train one model for generation and one for demand (both learn across
ALL houses at once -> the model generalizes to new houses too, which is
what you'd want in a real digital-twin platform).

Outputs:
    - trained models (joblib)
    - held-out test metrics (MAE, RMSE, R^2)
    - predictions_test.csv (used later by the routing optimizer / dashboard)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("neighborhood_timeseries.csv", parse_dates=["timestamp"])
houses = pd.read_csv("houses.csv")

df = df.merge(houses[["house_id", "panel_kw", "household_size", "profile"]], on="house_id")
df = pd.get_dummies(df, columns=["profile"], prefix="prof")

# lag features (per house, sorted by time)
df = df.sort_values(["house_id", "timestamp"])
for col in ["generation_kwh", "demand_kwh"]:
    df[f"{col}_lag1"] = df.groupby("house_id")[col].shift(1)
    df[f"{col}_lag24"] = df.groupby("house_id")[col].shift(24)

df = df.dropna().reset_index(drop=True)

feature_cols = (
    ["hour", "dow", "cloud_cover", "temp_c", "panel_kw", "household_size"]
    + [c for c in df.columns if c.startswith("prof_")]
    + ["generation_kwh_lag1", "generation_kwh_lag24", "demand_kwh_lag1", "demand_kwh_lag24"]
)

X = df[feature_cols]

# time-based split (train on first 30 days, test on remaining ~13 days) -
# this is more honest than a random split for a time-series forecasting task
cutoff = df["timestamp"].quantile(0.70)
train_mask = df["timestamp"] <= cutoff
test_mask = ~train_mask

results = {}
models = {}
for target in ["generation_kwh", "demand_kwh"]:
    y = df[target]
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = GradientBoostingRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.06,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    results[target] = {"MAE_kWh": round(mae, 3), "RMSE_kWh": round(rmse, 3), "R2": round(r2, 3)}
    models[target] = model
    joblib.dump(model, f"model_{target}.joblib")

    print(f"\n[{target}] test-set performance ({test_mask.sum()} hourly obs, "
          f"{df.loc[test_mask,'house_id'].nunique()} houses):")
    for k, v in results[target].items():
        print(f"   {k}: {v}")

# Save predictions for the test period -> feeds the routing optimizer + dashboard
out = df.loc[test_mask, ["house_id", "timestamp", "hour", "generation_kwh", "demand_kwh"]].copy()
out["pred_generation_kwh"] = np.clip(models["generation_kwh"].predict(X[test_mask]), 0, None)
out["pred_demand_kwh"] = np.clip(models["demand_kwh"].predict(X[test_mask]), 0, None)
out.to_csv("predictions_test.csv", index=False)

# feature importance (nice for a hackathon slide: "what drives the model")
for target, model in models.items():
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(6)
    print(f"\nTop features for {target}:")
    print(imp.round(3))

import json
with open("model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: model_generation_kwh.joblib, model_demand_kwh.joblib, "
      "predictions_test.csv, model_metrics.json")
