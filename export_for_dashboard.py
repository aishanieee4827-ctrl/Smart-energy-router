import pandas as pd
import json

houses = pd.read_csv("houses.csv")
preds = pd.read_csv("predictions_test.csv", parse_dates=["timestamp"])
routing = pd.read_csv("routing_results.csv", parse_dates=["timestamp"])
flows = pd.read_csv("routing_flows.csv", parse_dates=["timestamp"])
metrics = json.load(open("model_metrics.json"))

# pick 2 representative consecutive days that include some overlap (peer-routing) hours
day1, day2 = pd.Timestamp("2026-05-05"), pd.Timestamp("2026-05-06")
mask = (preds.timestamp >= day1) & (preds.timestamp < day2 + pd.Timedelta(days=1))
p = preds[mask].copy()
r = routing[(routing.timestamp >= day1) & (routing.timestamp < day2 + pd.Timedelta(days=1))].copy()
f = flows[(flows.timestamp >= day1) & (flows.timestamp < day2 + pd.Timedelta(days=1))].copy()

houses_out = houses[["house_id", "x", "y", "panel_kw", "household_size", "profile", "has_battery"]].to_dict("records")

hours_out = []
for ts, grp in p.groupby("timestamp"):
    hour_flows = f[f.timestamp == ts][["from", "to", "kwh", "via"]].to_dict("records")
    hour_metrics = r[r.timestamp == ts].iloc[0] if (r.timestamp == ts).any() else None
    hours_out.append({
        "ts": ts.isoformat(),
        "houses": {
            row.house_id: {
                "gen": round(row.generation_kwh, 3),
                "demand": round(row.demand_kwh, 3),
                "pred_gen": round(row.pred_generation_kwh, 3),
                "pred_demand": round(row.pred_demand_kwh, 3),
            } for row in grp.itertuples()
        },
        "flows": hour_flows,
        "baseline_loss": round(float(hour_metrics.baseline_loss), 4) if hour_metrics is not None else 0,
        "smart_loss": round(float(hour_metrics.smart_loss), 4) if hour_metrics is not None else 0,
        "curtailed_baseline": round(float(hour_metrics.curtailed_baseline), 4) if hour_metrics is not None else 0,
        "curtailed_smart": round(float(hour_metrics.curtailed_smart), 4) if hour_metrics is not None else 0,
    })

# full-period aggregate stats (all 316 test hours) for the headline metrics
total_baseline_loss = routing["baseline_loss"].sum()
total_smart_loss = routing["smart_loss"].sum()
total_curtailed_baseline = routing["curtailed_baseline"].sum()
total_curtailed_smart = routing["curtailed_smart"].sum()
total_surplus = routing["total_surplus"].sum()
total_deficit = routing["total_deficit"].sum()
overlap = routing[(routing.total_surplus > 0.1) & (routing.total_deficit > 0.1)]
overlap_reduction = (1 - overlap.smart_loss.sum() / overlap.baseline_loss.sum()) * 100
peer_kwh = flows[flows.via == "peer"].kwh.sum()

summary = {
    "test_period_hours": int(len(routing)),
    "test_period_days": int(len(routing) / 24),
    "n_houses": int(len(houses)),
    "total_surplus_kwh": round(float(total_surplus), 1),
    "total_deficit_kwh": round(float(total_deficit), 1),
    "loss_baseline_kwh": round(float(total_baseline_loss), 1),
    "loss_smart_kwh": round(float(total_smart_loss), 1),
    "loss_reduction_pct": round(float((1 - total_smart_loss / total_baseline_loss) * 100), 1),
    "curtailed_baseline_kwh": round(float(total_curtailed_baseline), 1),
    "curtailed_smart_kwh": round(float(total_curtailed_smart), 1),
    "overlap_hours": int(len(overlap)),
    "overlap_loss_reduction_pct": round(float(overlap_reduction), 1),
    "peer_to_peer_kwh_routed": round(float(peer_kwh), 1),
    "model_metrics": metrics,
}

out = {"houses": houses_out, "hours": hours_out, "summary": summary}
with open("dashboard_data.json", "w") as fp:
    json.dump(out, fp)

print("summary:", json.dumps(summary, indent=2))
print(f"\nhours embedded: {len(hours_out)}  |  json size: ", end="")
import os
print(f"{os.path.getsize('dashboard_data.json')/1024:.1f} KB")
