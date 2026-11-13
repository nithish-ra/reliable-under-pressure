"""Step 5: robustness of the top-10 lists.

Variants: base | leverage cut at top 35% / top 15% | IPL only | T20I only | no associate-nation matches
Also: split-half reliability of pressure WPA (random halves of matches) and a stability table for base top 10.
Run: python analysis/05_robustness.py   -> out/stability_bat.csv, out/stability_bowl.csv
"""
import os
import sys
import importlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R = importlib.import_module("04_ratings")
b, OUT = R.b, R.OUT
DATA = r"C:\Users\sivar\OneDrive\Desktop\RR\data\Data Pack"

# --- alternative pressure flags (leverage cut-offs) --------------------------
lg = b[b.legal]
contested = b.wp_before.between(0.2, 0.8)
shape = ((b.innings == 2) | (b.phase == "Death") | (b.wkts_before >= 4)) & ((b.phase != "PP") | (b.wkts_before >= 3))
for tag, q in [("p_q65", 0.65), ("p_q85", 0.85)]:
    b[tag] = (b.leverage >= lg.leverage.quantile(q)) & contested & shape

# --- associate-nation filter --------------------------------------------------
FULL = {"India", "Australia", "England", "South Africa", "New Zealand", "Pakistan", "West Indies", "Sri Lanka",
        "Bangladesh", "Afghanistan", "Ireland", "Zimbabwe"}
m = pd.read_csv(os.path.join(DATA, "matches.csv"))
c = pd.read_csv(os.path.join(DATA, "competitions.csv"))
m = m.merge(c[["comp_id", "match_format", "comp_name"]], on="comp_id")
bad = m[((m.match_format == "T20I") & ~(m.home_team.isin(FULL) & m.away_team.isin(FULL)))
        | ((m.match_format == "T20") & ~m.comp_name.str.startswith("IPL"))].match_id  # associates + A-team events
print("T20I matches involving associate sides dropped in 'no_assoc':", len(bad))

variants = {
    "base": (b, "high_pressure", {}),
    "lev_top35": (b, "p_q65", {}),
    "lev_top15": (b, "p_q85", {"min_balls": 40, "min_matches": 8}),
    "ipl_only": (b[b.is_ipl], "high_pressure", {"min_balls": 40, "min_matches": 8}),
    "t20i_only": (b[~b.is_ipl], "high_pressure", {"min_balls": 40, "min_matches": 8}),
    "no_assoc": (b[~b.match_id.isin(bad)], "high_pressure", {}),
}

res = {"bat": {}, "bowl": {}}
for role in ("bat", "bowl"):
    for name, (data, flag, kw) in variants.items():
        el = R.rate(role, data=data, flag=flag, save=False, verbose=False, **kw)
        res[role][name] = el["rank"]
        print(f"{role:4s} {name:10s} eligible={len(el)}")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
for role in ("bat", "bowl"):
    t = pd.DataFrame(res[role])
    base = pd.read_csv(os.path.join(OUT, "batters.csv" if role == "bat" else "bowlers.csv"), index_col="pid")
    t["name"] = base["name"]
    top = t[t.base <= 20].copy()
    vcols = [v for v in variants if v != "base"]
    top["in_top10"] = (top[vcols] <= 10).sum(1)
    top["in_top15"] = (top[vcols] <= 15).sum(1)
    top["worst"] = top[vcols].max(1)
    top = top.sort_values("base")
    print(f"\n=== {role}: ranks under each variant (NaN = not eligible); in_top10 counts of {len(vcols)} variants")
    print(top[["name", "base"] + vcols + ["in_top10", "in_top15", "worst"]].to_string())
    top.to_csv(os.path.join(OUT, f"stability_{role}.csv"))

# --- split-half reliability ---------------------------------------------------
rng = np.random.default_rng(5)
for role in ("bat", "bowl"):
    d = R.side_frame(role, b)
    hp = d[d.high_pressure]
    mids = hp.match_id.unique()
    cors = []
    for _ in range(30):
        half = set(rng.choice(mids, len(mids) // 2, replace=False))
        a = hp[hp.match_id.isin(half)].groupby("pid").val.agg(["mean", "size"])
        z = hp[~hp.match_id.isin(half)].groupby("pid").val.agg(["mean", "size"])
        j = a.join(z, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j.size_a >= 25) & (j.size_b >= 25)]
        cors.append(j.mean_a.corr(j.mean_b, method="spearman"))
    print(f"\nsplit-half Spearman reliability of pressure WPA/ball [{role}]: "
          f"mean={np.mean(cors):.2f} (range {np.min(cors):.2f}..{np.max(cors):.2f}) over {len(j)} players")
