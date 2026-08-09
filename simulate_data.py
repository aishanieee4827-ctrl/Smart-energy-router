"""
simulate_data.py
-----------------
Generates a synthetic but physically-plausible dataset for a neighborhood
"digital twin": N houses, each with a solar array of varying size, laid out
on a simple street-grid topology (so we have real distances between houses
for transmission-loss calculations).

For every house and every hour over `n_days`, we simulate:
    - weather (cloud cover 0-1, ambient temp C)
    - solar generation (kWh) - depends on panel size, time of day, cloud cover, temp
    - demand (kWh) - depends on household size/profile, time of day, temp (AC/heating), day-of-week

Output: neighborhood.csv, houses.csv (metadata + x,y grid coords)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_HOUSES = 24
N_DAYS = 45          # 30 days train, ~2 weeks test/demo
HOURS = N_DAYS * 24
GRID_SIDE = 6         # 6x6 street grid -> 24 lots (4 corners unused/park)

# ---------------------------------------------------------------------------
# 1. House metadata: position on a street grid (meters) + panel/household size
# ---------------------------------------------------------------------------
def build_houses(n=N_HOUSES):
    coords = []
    spacing = 40  # meters between adjacent lots
    idx = 0
    for row in range(GRID_SIDE):
        for col in range(GRID_SIDE):
            if len(coords) >= n:
                break
            coords.append((col * spacing, row * spacing))
    houses = []
    profiles = ["family", "retiree", "wfh_professional", "shift_worker"]
    for i, (x, y) in enumerate(coords):
        panel_kw = RNG.choice([3.0, 4.5, 6.0, 8.0], p=[0.3, 0.35, 0.25, 0.1])
        household_size = RNG.integers(1, 6)
        profile = RNG.choice(profiles)
        has_battery = bool(RNG.random() < 0.15)  # small existing battery (rare)
        houses.append({
            "house_id": f"H{i+1:02d}",
            "x": x, "y": y,
            "panel_kw": panel_kw,
            "household_size": household_size,
            "profile": profile,
            "has_battery": has_battery,
            "battery_kwh": round(RNG.uniform(5, 10), 1) if has_battery else 0.0,
        })
    return pd.DataFrame(houses)


# ---------------------------------------------------------------------------
# 2. Weather: shared across the neighborhood (it's small), varies by day
# ---------------------------------------------------------------------------
def build_weather(hours=HOURS):
    ts = pd.date_range("2026-04-01", periods=hours, freq="h")
    days = hours // 24
    # daily cloud "regime" with autocorrelation (weather persists across days)
    daily_cloud_base = np.clip(RNG.normal(0.35, 0.25, days), 0, 0.95)
    # smooth it a bit so consecutive days are correlated (simple AR)
    for i in range(1, days):
        daily_cloud_base[i] = 0.6 * daily_cloud_base[i - 1] + 0.4 * daily_cloud_base[i]
    cloud_cover = np.repeat(daily_cloud_base, 24) + RNG.normal(0, 0.05, hours)
    cloud_cover = np.clip(cloud_cover, 0, 1)

    hour_of_day = np.array([t.hour for t in ts])
    day_of_year = np.array([t.dayofyear for t in ts])
    # seasonal + diurnal temperature curve
    seasonal = 18 + 6 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    diurnal = 6 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    temp = seasonal + diurnal + RNG.normal(0, 1.5, hours)

    return pd.DataFrame({
        "timestamp": ts,
        "hour": hour_of_day,
        "dow": np.array([t.dayofweek for t in ts]),
        "cloud_cover": cloud_cover,
        "temp_c": temp,
    })


# ---------------------------------------------------------------------------
# 3. Solar generation model (per house, per hour)
# ---------------------------------------------------------------------------
def solar_irradiance_factor(hour):
    # bell curve centered at 13:00, zero before 6am / after 20:00
    if hour < 6 or hour > 20:
        return 0.0
    return max(0.0, np.sin(np.pi * (hour - 6) / 14)) ** 1.3


def gen_generation(weather, houses):
    rows = []
    irr = weather["hour"].apply(solar_irradiance_factor).values
    for _, h in houses.iterrows():
        base = h["panel_kw"] * irr
        cloud_attenuation = (1 - 0.75 * weather["cloud_cover"].values)
        # slight panel efficiency loss in extreme heat
        temp_derate = 1 - np.clip((weather["temp_c"].values - 30) * 0.004, 0, 0.1)
        noise = RNG.normal(1.0, 0.03, len(weather))
        gen_kwh = np.clip(base * cloud_attenuation * temp_derate * noise, 0, None)
        rows.append(gen_kwh)
    return np.array(rows)  # shape (n_houses, hours)


# ---------------------------------------------------------------------------
# 4. Demand model (per house, per hour) - profile driven
# ---------------------------------------------------------------------------
PROFILE_CURVES = {
    # 24-hour base multiplier shapes (roughly normalized, will be scaled by household size)
    "family":            [.3,.25,.2,.2,.25,.4,.7,.9,.6,.4,.35,.4,.45,.4,.4,.45,.6,.9,1.3,1.4,1.2,.9,.6,.4],
    "retiree":           [.4,.35,.3,.3,.35,.4,.5,.6,.65,.6,.55,.6,.65,.6,.55,.55,.6,.7,.8,.85,.75,.65,.55,.45],
    "wfh_professional":  [.3,.25,.2,.2,.25,.3,.4,.6,.7,.75,.75,.8,.8,.75,.75,.75,.7,.75,.9,1.0,.85,.65,.5,.4],
    "shift_worker":      [.35,.3,.3,.35,.4,.5,.5,.45,.4,.4,.7,.75,.7,.5,.45,.5,.55,.6,.7,.75,.8,.7,.5,.4],
}

def gen_demand(weather, houses):
    rows = []
    hours = weather["hour"].values
    dow = weather["dow"].values
    temp = weather["temp_c"].values
    for _, h in houses.iterrows():
        curve = np.array(PROFILE_CURVES[h["profile"]])
        base_shape = curve[hours]
        weekend_boost = np.where(dow >= 5, 1.15, 1.0)
        # HVAC load: extra demand when hot (AC) or cold (heating)
        hvac = 0.05 * np.clip(temp - 24, 0, None) + 0.04 * np.clip(10 - temp, 0, None)
        size_scale = 0.5 + 0.35 * h["household_size"]
        noise = RNG.normal(1.0, 0.06, len(weather))
        demand_kwh = np.clip((base_shape * size_scale + hvac) * weekend_boost * noise, 0.05, None)
        rows.append(demand_kwh)
    return np.array(rows)


def main():
    houses = build_houses()
    weather = build_weather()
    generation = gen_generation(weather, houses)   # (n_houses, hours)
    demand = gen_demand(weather, houses)           # (n_houses, hours)

    records = []
    for i, house_id in enumerate(houses["house_id"]):
        for t in range(len(weather)):
            records.append({
                "house_id": house_id,
                "timestamp": weather["timestamp"].iloc[t],
                "hour": weather["hour"].iloc[t],
                "dow": weather["dow"].iloc[t],
                "cloud_cover": weather["cloud_cover"].iloc[t],
                "temp_c": weather["temp_c"].iloc[t],
                "generation_kwh": generation[i, t],
                "demand_kwh": demand[i, t],
            })
    df = pd.DataFrame(records)
    df.to_csv("neighborhood_timeseries.csv", index=False)
    houses.to_csv("houses.csv", index=False)
    print(f"Saved neighborhood_timeseries.csv: {df.shape}")
    print(f"Saved houses.csv: {houses.shape}")
    print(df.groupby("house_id")[["generation_kwh", "demand_kwh"]].mean().round(2).head())


if __name__ == "__main__":
    main()
