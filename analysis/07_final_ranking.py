"""Step 7: final ratings, tuned and bootstrapped.

Value per ball (runs units, batter view; bowler = mirror image):
   v = (runs - E[runs]) - LAMBDA * (dismissal - E[dismissal])
   E[.] = league mean for the same innings x phase x match-tier (IPL / T20I-full / T20I-associate) x pressure flag.
Rating (M5) = (sum of v on pressure balls + K_BASE * baseline) / (n_pressure + K_BASE)
   baseline = player's own ordinary-ball mean v, itself shrunk to the league mean with K_ZERO pseudo-balls.
K_BASE / K_ZERO are chosen by out-of-sample prediction of held-out pressure performance (random match halves).
Uncertainty: bootstrap over matches -> P(top 10), 10th-90th percentile rank.
Outputs: out/final_batters.csv, out/final_bowlers.csv, out/tuning_grid.csv
"""
import os
import sys
import importlib
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R = importlib.import_module("04_ratings")
b, OUT = R.b, R.OUT
DATA = r"C:\Users\sivar\OneDrive\Desktop\RR\data\Data Pack"

MIN_P_BALLS, MIN_P_MATCHES, MIN_O_BALLS = 60, 10, 100
BOOT = 400

# ---- match tiers --------------------------------------------------------------
FULL = {"India", "Australia", "England", "South Africa", "New Zealand", "Pakistan", "West Indies", "Sri Lanka",
        "Bangladesh", "Afghanistan", "Ireland", "Zimbabwe"}
m = pd.read_csv(os.path.join(DATA, "matches.csv")).merge(
    pd.read_csv(os.path.join(DATA, "competitions.csv"))[["comp_id", "match_format"]], on="comp_id")
assoc = set(m[(m.match_format == "T20I") & ~(m.home_team.isin(FULL) & m.away_team.isin(FULL))].match_id)
b["tier"] = np.where(b.is_ipl, "IPL", np.where(b.match_format.eq("T20"), "Emerging",
                     np.where(b.match_id.isin(assoc), "T20I_assoc", "T20I_full")))

# ---- optional robustness variants (env VARIANT) -------------------------------
VARIANT = os.environ.get("VARIANT", "")
if VARIANT in ("q65", "q85"):  # looser / tighter leverage cut-off, same other conditions
    q = {"q65": 0.65, "q85": 0.85}[VARIANT]
    contested = b.wp_before.between(0.2, 0.8)
    shape = ((b.innings == 2) | (b.phase == "Death") | (b.wkts_before >= 4)) & ((b.phase != "PP") | (b.wkts_before >= 3))
    b["high_pressure"] = (b.leverage >= b[b.legal].leverage.quantile(q)) & contested & shape
elif VARIANT == "noassoc":
    b = b[b.tier.isin(["IPL", "T20I_full"])].copy()
elif VARIANT == "iplonly":
    b = b[b.tier == "IPL"].copy()
elif VARIANT == "t20ionly":
    b = b[b.tier.str.startswith("T20I")].copy()
if VARIANT in ("iplonly", "t20ionly"):
    MIN_P_BALLS, MIN_P_MATCHES, MIN_O_BALLS = 40, 8, 60

# ---- lambda (runs value of a wicket) -----------------------------------------
hpb = b[b.high_pressure & b.legal]
X = np.column_stack([np.ones(len(hpb)), hpb.team_runs.values, hpb.is_wicket.astype(float).values])
coef = np.linalg.lstsq(X, hpb.wpa.values, rcond=None)[0]
LAMBDA = float(-coef[2] / coef[1])
print(f"lambda = {LAMBDA:.2f} runs per wicket")


def prep(role):
    d = R.side_frame(role, b)
    if role == "bat":
        d["dis"] = (d.is_wicket & (d.dismissed_player_id == d.striker_id)).astype(float)
    else:
        d["dis"] = d.bowler_wkt.astype(float)
    key = d.cell + "_" + d.tier + "_" + d.high_pressure.astype(str)
    e_r, e_d = d.groupby(key).runs.transform("mean"), d.groupby(key).dis.transform("mean")
    if role == "bat":
        d["v"] = (d.runs - e_r) - LAMBDA * (d.dis - e_d)
    else:
        d["v"] = (e_r - d.runs) + LAMBDA * (d.dis - e_d)
    d["hp"] = d.high_pressure.astype(bool)
    return d


def agg_players(d):
    g = d.assign(ps=np.where(d.hp, d.v, 0.0), pn=d.hp.astype(float),
                 os_=np.where(~d.hp, d.v, 0.0), on=(~d.hp).astype(float))
    return g.groupby("pid")[["ps", "pn", "os_", "on"]].sum()


def m5(a, kb, kz):
    return (a.ps + kb * a.os_ / (a.on + kz)) / (a.pn + kb)


# ---- tuning: predict held-out pressure performance ----------------------------
KB = [0, 25, 50, 100, 200, 400, 800, 1600, 1e9]
KZ = [100, 300, 1000]
rng = np.random.default_rng(33)
data = {r: prep(r) for r in ("bat", "bowl")}
grid = []
rel_rows = []
for role, d in data.items():
    mids = d.match_id.unique()
    splits = []
    for _ in range(40):
        half = set(rng.choice(mids, len(mids) // 2, replace=False))
        splits.append((agg_players(d[d.match_id.isin(half)]), agg_players(d[~d.match_id.isin(half)])))
    # split-half repeatability of the final (tier-adjusted) metric: raw pressure value vs blended rating
    for kb in (0, 400):
        cs = []
        for A, B in splits:
            j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
            j = j[(j.pn_a >= 25) & (j.pn_b >= 25)]
            cs.append(m5(A.loc[j.index], kb, 300).corr(m5(B.loc[j.index], kb, 300), method="spearman"))
        rel_rows.append({"role": role, "k_base": kb, "split_half": float(np.mean(cs))})
    for kb in KB:
        for kz in KZ:
            cs = []
            for A, B in splits:
                j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
                j = j[(j.pn_a >= 25) & (j.pn_b >= 25)]
                pred = m5(A.loc[j.index], kb, kz)
                cs.append(pred.corr(j.ps_b / j.pn_b, method="spearman"))
            grid.append({"role": role, "k_base": kb, "k_zero": kz, "heldout_corr": np.mean(cs)})
grid = pd.DataFrame(grid)
sfx = f"_{VARIANT}" if VARIANT else ""
grid.to_csv(os.path.join(OUT, f"tuning_grid{sfx}.csv"), index=False)
pd.DataFrame(rel_rows).to_csv(os.path.join(OUT, f"split_half_final{sfx}.csv"), index=False)
print(pd.DataFrame(rel_rows).round(3).to_string(index=False))
piv = grid.groupby(["k_base", "k_zero"]).heldout_corr.mean().unstack()
print("\nheld-out predictive Spearman (mean of bat & bowl); rows k_base (0 = raw pressure only, 1e9 = baseline only)")
print(piv.round(3).to_string())
for role in ("bat", "bowl"):
    print(role, grid[grid.role == role].pivot(index="k_base", columns="k_zero", values="heldout_corr").round(3)
          .to_string(), sep="\n")

fin = grid[grid.k_base < 1e8].groupby(["k_base", "k_zero"]).heldout_corr.mean()
K_BASE, K_ZERO = fin.idxmax()
if os.environ.get("KB"):  # manual override for sensitivity runs: KB=400 KZ=300
    K_BASE, K_ZERO = float(os.environ["KB"]), float(os.environ["KZ"])
print(f"\nchosen (finite, pressure-weighted) K_BASE={K_BASE}, K_ZERO={K_ZERO}; "
      f"baseline-only best = {piv.loc[1e9].max():.3f}, grid max = {fin.max():.3f}")
TAG = os.environ.get("TAG", "")

# ---- final ratings + bootstrap -------------------------------------------------
for role, d in data.items():
    hp = d[d.hp]
    tot = agg_players(d)
    pm = hp.groupby("pid").agg(p_matches=("match_id", "nunique"))
    tot = tot.join(pm)
    tot["m5"] = m5(tot, K_BASE, K_ZERO)
    el = tot[(tot.pn >= MIN_P_BALLS) & (tot.p_matches >= MIN_P_MATCHES) & (tot.on >= MIN_O_BALLS)].copy()

    # bootstrap over matches
    pm_ = d.assign(ps=np.where(d.hp, d.v, 0.0), pn=d.hp.astype(float), os_=np.where(~d.hp, d.v, 0.0),
                   on=(~d.hp).astype(float))
    pm_ = pm_[pm_.pid.isin(el.index)].groupby(["pid", "match_id"])[["ps", "pn", "os_", "on"]].sum().reset_index()
    pid_ix = {p: i for i, p in enumerate(el.index)}
    mid_ix = {x: i for i, x in enumerate(d.match_id.unique())}
    ri, ci = pm_.pid.map(pid_ix).values, pm_.match_id.map(mid_ix).values
    S = {c: sparse.csr_matrix((pm_[c].values, (ri, ci)), shape=(len(el), len(mid_ix))) for c in ["ps", "pn", "os_", "on"]}
    C = np.random.default_rng(99).multinomial(len(mid_ix), np.ones(len(mid_ix)) / len(mid_ix), size=BOOT).T.astype(float)
    T = {c: S[c] @ C for c in S}
    boot = (T["ps"] + K_BASE * T["os_"] / (T["on"] + K_ZERO)) / (T["pn"] + K_BASE)
    ranks = (-boot).argsort(0).argsort(0) + 1
    el["p_top10"] = (ranks <= 10).mean(1)
    el["rank_p10"], el["rank_p90"] = np.percentile(ranks, 10, axis=1), np.percentile(ranks, 90, axis=1)

    # explainability columns
    h = hp.groupby("pid")
    el["pressure_v"] = el.ps / el.pn
    el["baseline_v"] = el.os_ / el.on
    el["lift"] = el.pressure_v - el.baseline_v
    mm = hp.groupby(["pid", "match_id"]).agg(n=("v", "size"), s=("v", "sum")).reset_index()
    mm = mm[mm.n >= 4]
    el["consist"] = mm.groupby("pid").s.agg(lambda x: (x > 0).mean()).reindex(el.index)
    if role == "bat":
        el["p_sr"] = (h.runs.sum() / h.bat_faced.sum() * 100).reindex(el.index)
        el["balls_per_dis"] = (h.size() / h.dis.sum().replace(0, np.nan)).reindex(el.index)
    else:
        el["p_econ"] = (h.runs.sum() / h.legal.sum() * 6).reindex(el.index)
        el["balls_per_wkt"] = (h.legal.sum() / h.dis.sum().replace(0, np.nan)).reindex(el.index)
    for tag, sel in [("ipl", hp.tier == "IPL"), ("t20i", hp.tier.str.startswith("T20I")), ("chase", hp.innings == 2),
                     ("bat1st", hp.innings == 1), ("mid", hp.phase == "Middle"), ("death", hp.phase == "Death")]:
        t = hp[sel].groupby("pid").v.agg(["mean", "size"])
        el[f"v_{tag}"] = t["mean"].reindex(el.index)
        el[f"n_{tag}"] = t["size"].reindex(el.index)
    names = d.groupby("pid").pname.agg(lambda x: x.value_counts().index[0])
    el["name"] = names.reindex(el.index)
    el = el.sort_values("m5", ascending=False)
    el.insert(0, "rank", np.arange(1, len(el) + 1))
    el.to_csv(os.path.join(OUT, f"final_{'batters' if role == 'bat' else 'bowlers'}{TAG}.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"\n=== FINAL {role.upper()} (eligible {len(el)}) ===")
    cols = ["rank", "name", "pn", "p_matches", "pressure_v", "baseline_v", "lift", "m5", "consist", "p_top10",
            "rank_p10", "rank_p90", "n_ipl", "n_t20i"]
    print(el[cols].head(20).round(3).to_string())
