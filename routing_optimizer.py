"""
routing_optimizer.py
---------------------
Given the ML-forecasted generation & demand for every house at a given hour,
decide how to route surplus solar energy across the neighborhood to minimize
distance-proportional transmission loss, using linear programming
(a classic transportation problem) solved with scipy.optimize.linprog (HiGHS).

Two scenarios are computed for every hour so we can show the improvement:

  BASELINE (today's grid): every surplus kWh is exported to the substation
  and every deficit kWh is imported from the substation. All energy travels
  the *full* distance to/from the substation -> more transmission loss, and
  surplus beyond what the substation/grid can usefully absorb is curtailed
  (wasted) up to a `substation_export_cap`.

  SMART (digital twin routing): the optimizer first tries to match nearby
  surplus houses directly to nearby deficit houses (short peer-to-peer
  distance = much lower loss), only falling back to the substation for
  energy that can't be matched locally.

Loss model: loss_fraction = ALPHA * distance_meters (simple linear line-loss
approximation, capped at 30%).
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ALPHA = 0.00035          # loss fraction per meter of line
SUBSTATION_XY = (100, 100)   # roughly the center of our 6x6 grid (spacing=40)
SUBSTATION_EXPORT_CAP_KWH = 3.0  # per-hour cap on how much surplus the substation/grid can absorb from the neighborhood before curtailment


def loss_frac(dist_m):
    return min(ALPHA * dist_m, 0.30)


def distance(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def solve_hour(surplus, deficit, house_coords):
    """
    surplus, deficit: dict house_id -> kWh (only positive entries)
    house_coords: dict house_id -> (x, y)
    Returns dict with baseline_loss, smart_loss, smart_curtailed, smart_unmet, flows
    """
    surplus_ids = list(surplus.keys())
    deficit_ids = list(deficit.keys())

    # ---------- BASELINE: everything via substation ----------
    baseline_export_loss = sum(
        min(v, SUBSTATION_EXPORT_CAP_KWH * v / max(sum(surplus.values()), 1e-9))  # proportional cap (simplified)
        for v in surplus.values()
    )
    # simpler & clearer: cap total exported surplus, rest is curtailed (wasted)
    total_surplus = sum(surplus.values())
    exported = min(total_surplus, SUBSTATION_EXPORT_CAP_KWH)
    curtailed_baseline = total_surplus - exported
    baseline_loss = 0.0
    for hid, v in surplus.items():
        share = (v / total_surplus) if total_surplus > 0 else 0
        exported_v = exported * share
        baseline_loss += exported_v * loss_frac(distance(house_coords[hid], SUBSTATION_XY))
    for hid, v in deficit.items():
        baseline_loss += v * loss_frac(distance(house_coords[hid], SUBSTATION_XY))

    # ---------- SMART: LP transportation problem (peer-to-peer + substation fallback) ----------
    # nodes: surplus houses (supply) -> deficit houses + substation (demand)
    #        substation (supply, unlimited-ish) -> deficit houses
    n_s, n_d = len(surplus_ids), len(deficit_ids)
    if n_s == 0 and n_d == 0:
        return dict(baseline_loss=0.0, smart_loss=0.0, curtailed_baseline=0.0,
                     curtailed_smart=0.0, unmet_smart=0.0, flows=[])

    # variables: x[i,j] house_i -> house_j  (n_s * n_d)
    #            xs[j]  substation -> house_j (n_d)
    #            xe[i]  house_i -> substation (n_s), capped by remaining substation export cap
    n_x = n_s * n_d
    n_xs = n_d
    n_xe = n_s
    n_vars = n_x + n_xs + n_xe

    # M is a large "reward" for delivering a kWh to a deficit house (via peer or
    # substation) - much bigger than any possible loss_frac (max 0.30), so the
    # LP's *primary* objective becomes "satisfy as much demand as possible",
    # and *among* equally-good delivery plans it picks the one with lowest
    # transmission loss (i.e. prefers short peer-to-peer hops over the distant
    # substation). xe (house -> substation, pure export/curtailment bookkeeping)
    # carries no delivery reward since it doesn't serve any deficit itself.
    M = 10.0
    loss_only = np.zeros(n_vars)   # pure loss fraction per unit, for REPORTING
    c = np.zeros(n_vars)           # LP objective (loss - delivery reward), for SOLVING
    for i, sid in enumerate(surplus_ids):
        for j, did in enumerate(deficit_ids):
            lf = loss_frac(distance(house_coords[sid], house_coords[did]))
            loss_only[i * n_d + j] = lf
            c[i * n_d + j] = lf - M
    for j, did in enumerate(deficit_ids):
        lf = loss_frac(distance(SUBSTATION_XY, house_coords[did]))
        loss_only[n_x + j] = lf
        c[n_x + j] = lf - M
    # xe (house -> substation export): tiny tie-breaking incentive so leftover
    # surplus that can't help any deficit still fills the substation's export
    # allowance (rather than the LP arbitrarily leaving it as idle slack).
    for i in range(n_s):
        c[n_x + n_xs + i] = -1e-6

    A_ub = []
    b_ub = []
    # supply constraints: sum_j x[i,j] + xe[i] <= surplus_i
    for i in range(n_s):
        row = np.zeros(n_vars)
        for j in range(n_d):
            row[i * n_d + j] = 1
        row[n_x + n_xs + i] = 1
        A_ub.append(row); b_ub.append(surplus[surplus_ids[i]])
    # demand constraints: sum_i x[i,j] + xs[j] <= deficit_j  (<=, unmet allowed but penalized heavily via cost... use eq with slack instead)
    for j in range(n_d):
        row = np.zeros(n_vars)
        for i in range(n_s):
            row[i * n_d + j] = 1
        row[n_x + j] = 1
        A_ub.append(row); b_ub.append(deficit[deficit_ids[j]])
    # substation export cap: sum_i xe[i] <= remaining cap
    row = np.zeros(n_vars)
    for i in range(n_s):
        row[n_x + n_xs + i] = 1
    A_ub.append(row); b_ub.append(SUBSTATION_EXPORT_CAP_KWH)

    bounds = [(0, None)] * n_vars
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")

    flows = []
    if res.success:
        x = res.x
        smart_loss = 0.0
        for i, sid in enumerate(surplus_ids):
            for j, did in enumerate(deficit_ids):
                v = x[i * n_d + j]
                if v > 1e-4:
                    smart_loss += v * loss_only[i * n_d + j]
                    flows.append({"from": sid, "to": did, "kwh": round(v, 4), "via": "peer"})
        for j, did in enumerate(deficit_ids):
            v = x[n_x + j]
            if v > 1e-4:
                smart_loss += v * loss_only[n_x + j]
                flows.append({"from": "SUBSTATION", "to": did, "kwh": round(v, 4), "via": "grid"})
        exported_smart = sum(x[n_x + n_xs + i] for i in range(n_s))
        curtailed_smart = total_surplus - exported_smart - sum(
            x[i * n_d + j] for i in range(n_s) for j in range(n_d)
        )
        met_demand = sum(x[i * n_d + j] for i in range(n_s) for j in range(n_d)) + sum(x[n_x + j] for j in range(n_d))
        unmet_smart = max(0.0, sum(deficit.values()) - met_demand)
    else:
        smart_loss, curtailed_smart, unmet_smart = baseline_loss, curtailed_baseline, 0.0

    return dict(
        baseline_loss=baseline_loss,
        smart_loss=smart_loss,
        curtailed_baseline=curtailed_baseline,
        curtailed_smart=max(0.0, curtailed_smart),
        unmet_smart=unmet_smart,
        flows=flows,
    )


def main():
    houses = pd.read_csv("houses.csv")
    coords = {row.house_id: (row.x, row.y) for row in houses.itertuples()}
    preds = pd.read_csv("predictions_test.csv", parse_dates=["timestamp"])

    hourly_results = []
    all_flows = []
    for ts, grp in preds.groupby("timestamp"):
        net = grp.set_index("house_id").apply(
            lambda r: r["pred_generation_kwh"] - r["pred_demand_kwh"], axis=1
        )
        surplus = {h: v for h, v in net.items() if v > 0.01}
        deficit = {h: -v for h, v in net.items() if v < -0.01}
        r = solve_hour(surplus, deficit, coords)
        r["timestamp"] = ts
        r["total_surplus"] = sum(surplus.values())
        r["total_deficit"] = sum(deficit.values())
        hourly_results.append(r)
        for f in r["flows"]:
            f["timestamp"] = ts
            all_flows.append(f)

    res_df = pd.DataFrame(hourly_results).drop(columns=["flows"])
    res_df.to_csv("routing_results.csv", index=False)
    pd.DataFrame(all_flows).to_csv("routing_flows.csv", index=False)

    total_baseline_loss = res_df["baseline_loss"].sum()
    total_smart_loss = res_df["smart_loss"].sum()
    total_curtailed_baseline = res_df["curtailed_baseline"].sum()
    total_curtailed_smart = res_df["curtailed_smart"].sum()
    total_surplus = res_df["total_surplus"].sum()

    print(f"Hours simulated: {len(res_df)}")
    print(f"Total surplus solar available: {total_surplus:.1f} kWh")
    print(f"\nTransmission loss  -> baseline: {total_baseline_loss:.1f} kWh | smart: {total_smart_loss:.1f} kWh "
          f"({(1 - total_smart_loss/total_baseline_loss)*100:.1f}% reduction)")
    print(f"Curtailed (wasted) -> baseline: {total_curtailed_baseline:.1f} kWh | smart: {total_curtailed_smart:.1f} kWh "
          f"({(1 - total_curtailed_smart/max(total_curtailed_baseline,1e-9))*100:.1f}% reduction)")
    print(f"Unmet demand (smart routing): {res_df['unmet_smart'].sum():.2f} kWh")
    print("\nSaved: routing_results.csv, routing_flows.csv")


if __name__ == "__main__":
    main()
