"""
load_real_solar_data.py
------------------------
Turns the real Kaggle "Solar Power Generation Data" CSVs into the exact same
neighborhood_timeseries.csv / houses.csv shape that forecasting_model.py and
routing_optimizer.py already expect - so NOTHING downstream needs to change.

WHAT THIS DOES, IN PLAIN ENGLISH:
  The dataset has one solar PLANT, but that plant has ~22 separate inverters
  (each with its own SOURCE_KEY). Each inverter is its own meter with its
  own slightly-different generation curve (different shading, wiring, panel
  wear) - which is actually perfect for us: we treat EACH INVERTER AS ONE
  "HOUSE" in the neighborhood. That gives us ~22 real, physically-distinct
  generation profiles instead of 1.

  Demand has no real data (explained earlier in the project) - we reuse the
  same synthetic profile-curve demand model from simulate_data.py, just
  wired to the REAL timestamps/hours from this dataset instead of made-up
  ones, so generation and demand are aligned in time.

HOW TO RUN:
  1. Put Plant_1_Generation_Data.csv, Plant_1_Weather_Sensor_Data.csv,
     Plant_2_Generation_Data.csv, Plant_2_Weather_Sensor_Data.csv in this
     same folder (Plant 2 is the better one - Plant 1's DC/AC readings have
     a known unit inconsistency in the original Kaggle dataset).
  2. python3 load_real_solar_data.py
  3. Produces: neighborhood_timeseries.csv, houses.csv - same shape as
     before, so forecasting_model.py runs on this with ZERO changes.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

PLANT = "Plant_2"  # Plant 2's readings are cleaner than Plant 1's in this dataset
GEN_FILE = f"{PLANT}_Generation_Data.csv"
WEATHER_FILE = f"{PLANT}_Weather_Sensor_Data.csv"

PROFILE_CURVES = {
    "family":            [.3,.25,.2,.2,.25,.4,.7,.9,.6,.4,.35,.4,.45,.4,.4,.45,.6,.9,1.3,1.4,1.2,.9,.6,.4],
    "retiree":           [.4,.35,.3,.3,.35,.4,.5,.6,.65,.6,.55,.6,.65,.6,.55,.55,.6,.7,.8,.85,.75,.65,.55,.45],
    "wfh_professional":  [.3,.25,.2,.2,.25,.3,.4,.6,.7,.75,.75,.8,.8,.75,.75,.75,.7,.75,.9,1.0,.85,.65,.5,.4],
    "shift_worker":      [.35,.3,.3,.35,.4,.5,.5,.45,.4,.4,.7,.75,.7,.5,.45,.5,.55,.6,.7,.75,.8,.7,.5,.4],
}
PROFILES = list(PROFILE_CURVES.keys())


def load_and_merge():
    gen = pd.read_csv(GEN_FILE)
    weather = pd.read_csv(WEATHER_FILE)

    # both files use DATE_TIME but in slightly different string formats in
    # the original Kaggle CSVs - parse loosely and let pandas figure it out
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=True, errors="coerce")
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")

    # weather is plant-level (one reading per timestamp), generation is
    # per-inverter (SOURCE_KEY) - merge weather onto every inverter row
    weather_small = weather[["DATE_TIME", "AMBIENT_TEMPERATURE", "IRRADIATION"]].drop_duplicates("DATE_TIME")
    df = gen.merge(weather_small, on="DATE_TIME", how="inner")

    # AC_POWER is in kW, readings are every 15 min -> kWh per interval = kW * 0.25
    df["generation_kwh_15min"] = df["AC_POWER"].clip(lower=0) * 0.25

    return df


def to_hourly_per_inverter(df):
    df["hour_bucket"] = df["DATE_TIME"].dt.floor("h")
    hourly = (
        df.groupby(["SOURCE_KEY", "hour_bucket"])
        .agg(generation_kwh=("generation_kwh_15min", "sum"),
             temp_c=("AMBIENT_TEMPERATURE", "mean"),
             irradiation=("IRRADIATION", "mean"))
        .reset_index()
        .rename(columns={"hour_bucket": "timestamp"})
    )
    return hourly


def build_houses(source_keys):
    """One 'house' per real inverter. panel_kw estimated from that inverter's
    own observed peak output (a reasonable proxy since we don't have a
    separate nameplate-capacity column in this dataset)."""
    coords, spacing, side = [], 40, 6
    for row in range(side):
        for col in range(side):
            if len(coords) >= len(source_keys):
                break
            coords.append((col * spacing, row * spacing))

    houses = []
    for i, (sk, (x, y)) in enumerate(zip(source_keys, coords)):
        profile = PROFILES[i % len(PROFILES)]  # cycle through so all 4 types appear
        household_size = int(RNG.integers(1, 6))
        has_battery = bool(RNG.random() < 0.15)
        houses.append({
            "house_id": f"H{i+1:02d}", "source_key": sk, "x": x, "y": y,
            "household_size": household_size, "profile": profile,
            "has_battery": has_battery, "battery_kwh": round(RNG.uniform(5, 10), 1) if has_battery else 0.0,
        })
    return pd.DataFrame(houses)


def synthetic_demand_for(df):
    """df needs columns: profile, household_size, hour, dow, temp_c. Returns an array of demand_kwh."""
    base_shape = df.apply(lambda r: PROFILE_CURVES[r["profile"]][int(r["hour"])], axis=1).values
    weekend_boost = np.where(df["dow"].values >= 5, 1.15, 1.0)
    temp = df["temp_c"].values
    hvac = 0.05 * np.clip(temp - 24, 0, None) + 0.04 * np.clip(10 - temp, 0, None)
    size_scale = 0.5 + 0.35 * df["household_size"].values
    noise = RNG.normal(1.0, 0.06, len(df))
    return np.clip((base_shape * size_scale + hvac) * weekend_boost * noise, 0.05, None)


def main():
    print("Loading & merging real Kaggle solar data...")
    raw = load_and_merge()
    hourly = to_hourly_per_inverter(raw)

    source_keys = sorted(hourly["source_key"].unique() if "source_key" in hourly else hourly["SOURCE_KEY"].unique())
    houses = build_houses(source_keys)
    print(f"Found {len(source_keys)} real inverters -> treating as {len(houses)} houses")

    # figure out per-inverter panel_kw proxy from observed peak generation
    peak_by_key = hourly.groupby("SOURCE_KEY")["generation_kwh"].max()
    houses["panel_kw"] = houses["source_key"].map(peak_by_key).round(1).fillna(peak_by_key.median())

    merged = hourly.merge(houses[["house_id", "source_key", "profile", "household_size"]],
                           left_on="SOURCE_KEY", right_on="source_key")
    merged["hour"] = merged["timestamp"].dt.hour
    merged["dow"] = merged["timestamp"].dt.dayofweek
    merged["cloud_cover"] = (1 - (merged["irradiation"] / merged["irradiation"].max().clip(min=0.01))).clip(0, 1)

    merged["demand_kwh"] = synthetic_demand_for(
        merged[["profile", "household_size", "hour", "dow"]].assign(temp_c=merged["temp_c"].fillna(25))
    )

    out = merged[["house_id", "timestamp", "hour", "dow", "cloud_cover", "temp_c",
                  "generation_kwh", "demand_kwh"]].copy()
    out = out.sort_values(["house_id", "timestamp"]).reset_index(drop=True)

    out.to_csv("neighborhood_timeseries.csv", index=False)
    houses[["house_id", "x", "y", "panel_kw", "household_size", "profile", "has_battery", "battery_kwh"]].to_csv(
        "houses.csv", index=False
    )

    print(f"\nSaved neighborhood_timeseries.csv: {out.shape}")
    print(f"Saved houses.csv: {houses.shape}")
    print(f"\nDate range: {out['timestamp'].min()} to {out['timestamp'].max()}")
    print(f"\nGeneration summary (real data):")
    print(out.groupby("house_id")["generation_kwh"].mean().describe().round(2))
    print("\nReady - run forecasting_model.py and routing_optimizer.py unchanged from here.")


if __name__ == "__main__":
    main()
