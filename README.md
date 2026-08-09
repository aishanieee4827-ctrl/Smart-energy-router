# GridWeave — Smart Energy Router

An AI digital twin that forecasts each home's solar generation and demand,
then computes how much a smarter peer-to-peer routing scheme could cut
transmission loss and wasted solar compared to today's grid.

## What it does

1. **Forecasts** next-hour solar generation and household demand per home,
   using a scikit-learn GradientBoostingRegressor trained on weather + house
   attributes + historical lag features.
2. **Routes** predicted surplus to predicted deficit homes every hour by
   solving a linear program (`scipy.optimize.linprog`) that minimizes
   distance-weighted transmission loss — preferring short peer-to-peer hops
   over the distant substation.
3. **Compares** this against a baseline where every home routes through the
   substation only (how today's grid effectively works), to quantify the
   improvement.

## What's real vs. modeled

- **Generation is real data** — from a real Indian solar plant (Kaggle
  "Solar Power Generation Data", 22 inverters), rescaled from utility-farm
  scale down to realistic residential rooftop panel sizes (3–8 kW), keeping
  the real weather-driven shape.
- **Demand is a calibrated synthetic model** — no public Indian household
  smart-meter dataset exists yet, so demand uses profile-driven curves
  (family / retiree / WFH / shift-worker).
- **Grid topology is synthetic** — a simplified street-grid layout, since
  real distribution-line distances aren't public.

## Results (9-day real-data test window, 22 homes)

| Metric | Value |
|---|---|
| Generation forecast R² | 0.84 |
| Demand forecast R² | 0.96 |
| Transmission loss reduction | 10.5% |
| Curtailed (wasted) solar reduction | 14.6% |
| Loss reduction in peer-match hours | 52.4% |

## Files

- `load_real_solar_data.py` — merges raw Kaggle generation + weather CSVs
  into one clean hourly table, rescaled to residential scale
- `forecasting_model.py` — trains the two GradientBoosting forecasting models
- `routing_optimizer.py` — the hourly linear-program routing engine
- `export_for_dashboard.py` — packages results for the live dashboard
- `simulate_data.py` — earlier synthetic-data version, kept for reference

## Tech stack

Python · pandas · scikit-learn · scipy (linprog) · HTML/CSS/JS dashboard

## Roadmap

- Real household smart-meter data (currently the main open gap)
- Battery storage layer to time-shift midday surplus into evening deficit
- Real distribution-grid topology via a utility/DISCOM partnership
