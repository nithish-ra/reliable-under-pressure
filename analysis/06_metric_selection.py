"""Step 6: choose the rating metric by repeatability.

Candidate per-ball values on high-pressure balls (batter view; bowler = mirror image):
  M1 WPA                          win probability added
  M2 RAE                          runs above state-cell expectation
  M3 DIS                          -(dismissal - expected)          (bowler: wickets - expected)
  M4 RAE+DIS                      M2 and M3 combined using lambda (runs value of a wicket, from regression of WPA)
  M5 shrunk-to-baseline           M4 on pressure balls shrunk towards the player's OWN ordinary-ball M4
Evaluation: split-half Spearman reliability (random halves of matches, 40 reps) and transfer
(ordinary-ball M4 in half A vs pressure M4 in half B).
Run: python analysis/06_metric_selection.py
"""
import os
import sys
import importlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R = importlib.import_module("04_ratings")
b, OUT = R.b, R.OUT

# ---- lambda: runs-equivalent of a wicket, from pressure balls ----------------
hpb = b[b.high_pressure & b.legal]
X = np.column_stack([np.ones(len(hpb)), hpb.team_runs.values, hpb.is_wicket.astype(float).values])
coef = np.linalg.lstsq(X, hpb.wpa.values, rcond=None)[0]
LAMBDA = float(-coef[2] / coef[1])
print(f"WPA regression: per run {coef[1]:+.4f}, per wicket {coef[2]:+.4f}  => lambda = {LAMBDA:.2f} runs per wicket")


def prep(role):
    d = R.side_frame(role, b)
    if role == "bat":
        d["dis"] = (d.is_wicket & (d.dismissed_player_id == d.striker_id)).astype(float)
        sgn = 1.0
    else:
        d["dis"] = d.bowler_wkt.astype(float)
        sgn = -1.0
    # expected runs / dismissals from league cell x pressure flag
    cell = d.cell + "_" + d.high_pressure.astype(str)
    d["e_r"] = d.groupby(cell).runs.transform("mean")
    d["e_d"] = d.groupby(cell).dis.transform("mean")
    if role == "bat":
        d["m2"] = d.runs - d.e_r
        d["m3"] = -(d.dis - d.e_d)
    else:
        d["m2"] = d.e_r - d.runs
        d["m3"] = d.dis - d.e_d
    # M4 in runs units: batter: runs above exp minus lambda*extra dismissals; bowler mirror
    d["m4"] = d.m2 + LAMBDA * d.m3
    d["m1"] = d.val
    return d


def per_player(d, k_base=150, k_zero=300):
    hp, od = d[d.high_pressure], d[~d.high_pressure]
    p = hp.groupby("pid").agg(n=("m1", "size"), m1=("m1", "mean"), m2=("m2", "mean"), m3=("m3", "mean"),
                               m4=("m4", "mean"), s4=("m4", "sum"))
    o = od.groupby("pid").agg(on=("m4", "size"), os=("m4", "sum"))
    p = p.join(o, how="left").fillna({"on": 0, "os": 0})
    base = p.os / (p.on + k_zero)  # ordinary-ball level, shrunk to league mean 0
    p["m5"] = (p.s4 + k_base * base) / (p.n + k_base)
    p["base"] = p.os / p.on.replace(0, np.nan)
    return p


rng = np.random.default_rng(21)
for role in ("bat", "bowl"):
    d = prep(role)
    mids = d[d.high_pressure].match_id.unique()
    all_mids = d.match_id.unique()
    rows, tr = [], []
    for _ in range(40):
        half = set(rng.choice(all_mids, len(all_mids) // 2, replace=False))
        A = per_player(d[d.match_id.isin(half)])
        B = per_player(d[~d.match_id.isin(half)])
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j.n_a >= 25) & (j.n_b >= 25)]
        rows.append({m: j[f"{m}_a"].corr(j[f"{m}_b"], method="spearman") for m in ["m1", "m2", "m3", "m4", "m5"]})
        jj = j[(j.on_a >= 100)]
        tr.append(jj.base_a.corr(jj.m4_b, method="spearman"))
    r = pd.DataFrame(rows)
    print(f"\n[{role}] split-half Spearman reliability over {len(j)} players (mean over 40 random splits)")
    print(r.mean().round(3).to_string())
    print(f"[{role}] transfer: ordinary-ball M4 (half A) -> pressure M4 (half B): {np.mean(tr):.3f}")
