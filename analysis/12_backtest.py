"""Step 12: walk-forward backtest.
Rate players using ONLY matches before a cut-off date, then measure what they did on pressure balls AFTER it.
Compares three ratings: raw pressure record (K=0), our blend (K=400), baseline only (ordinary-ball level).
Outputs out/backtest.csv and prints the table.  Run: python analysis/12_backtest.py
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
KB, KZ = 400, 300

FULL = {"India", "Australia", "England", "South Africa", "New Zealand", "Pakistan", "West Indies", "Sri Lanka",
        "Bangladesh", "Afghanistan", "Ireland", "Zimbabwe"}
m = pd.read_csv(os.path.join(DATA, "matches.csv")).merge(
    pd.read_csv(os.path.join(DATA, "competitions.csv"))[["comp_id", "match_format"]], on="comp_id")
assoc = set(m[(m.match_format == "T20I") & ~(m.home_team.isin(FULL) & m.away_team.isin(FULL))].match_id)
b["tier"] = np.where(b.is_ipl, "IPL", np.where(b.match_format.eq("T20"), "Emerging",
                     np.where(b.match_id.isin(assoc), "T20I_assoc", "T20I_full")))
hpb = b[b.high_pressure & b.legal]
X = np.column_stack([np.ones(len(hpb)), hpb.team_runs.values, hpb.is_wicket.astype(float).values])
coef = np.linalg.lstsq(X, hpb.wpa.values, rcond=None)[0]
LAMBDA = float(-coef[2] / coef[1])


def prep(role):
    d = R.side_frame(role, b)
    d["dis"] = (d.is_wicket & (d.dismissed_player_id == d.striker_id)).astype(float) if role == "bat" else d.bowler_wkt.astype(float)
    key = d.cell + "_" + d.tier + "_" + d.high_pressure.astype(str)
    e_r, e_d = d.groupby(key).runs.transform("mean"), d.groupby(key).dis.transform("mean")
    d["v"] = ((d.runs - e_r) - LAMBDA * (d.dis - e_d)) if role == "bat" else ((e_r - d.runs) + LAMBDA * (d.dis - e_d))
    d["hp"] = d.high_pressure.astype(bool)
    d["date"] = pd.to_datetime(d.match_date)
    return d


def agg(d):
    g = d.assign(ps=np.where(d.hp, d.v, 0.0), pn=d.hp.astype(float), os_=np.where(~d.hp, d.v, 0.0), on=(~d.hp).astype(float))
    return g.groupby("pid")[["ps", "pn", "os_", "on"]].sum()


def rating(a, kb):
    base = a.os_ / (a.on + KZ)
    return (a.ps + kb * base) / (a.pn + kb) if kb < 1e8 else base


rows = []
for cut in ("2024-01-01", "2025-01-01"):
    for role in ("bat", "bowl"):
        d = prep(role)
        tr, te = agg(d[d.date < cut]), agg(d[d.date >= cut])
        el = tr[(tr.pn >= 40) & (tr.on >= 100)]
        te = te[te.pn >= 30]
        j = el.join(te, lsuffix="_tr", rsuffix="_te", how="inner")
        tgt = j.ps_te / j.pn_te
        league_te = te.ps.sum() / te.pn.sum()
        for name, kb in (("pressure record only (K=0)", 0), ("our blend (K=400)", KB), ("ordinary level only", 1e9)):
            r = rating(el.loc[j.index], kb)
            rho = r.corr(tgt, method="spearman")
            top = r.sort_values(ascending=False).head(10).index
            sel = j.loc[top]
            pooled = sel.ps_te.sum() / sel.pn_te.sum()
            allp = j.ps_te.sum() / j.pn_te.sum()
            rows.append({"cutoff": cut, "role": role, "rating": name, "players": len(j), "spearman_next": rho,
                         "top10_next_v": pooled * 100, "all_rated_next_v": allp * 100, "top10_n_test_balls": sel.pn_te.sum()})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "backtest.csv"), index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
