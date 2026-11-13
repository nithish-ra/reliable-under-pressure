"""Step 3: compare pressure definitions and save the chosen flag.

A: top-quartile leverage only
B: A and contested game (20% <= win prob <= 80%)
C: B and (chase, or 1st-innings death, or 1st-innings collapse with >=4 wickets down)
Input: out/state_wp.csv.gz   Output: out/state_final.csv.gz
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
b = pd.read_csv(os.path.join(OUT, "state_wp.csv.gz"), low_memory=False)
lg = b[b.legal]
q75 = lg.leverage.quantile(0.75)

contested = b.wp_before.between(0.2, 0.8)
base = b.leverage >= q75
b["pA"] = base
b["pB"] = base & contested
b["pC"] = b.pB & ((b.innings == 2) | (b.phase == "Death") | (b.wkts_before >= 4))
# D: early-innings balls are only 'pressure' after a collapse (>=3 down in the powerplay)
b["pD"] = b.pC & ((b.phase != "PP") | (b.wkts_before >= 3))


def summ(d):
    return pd.Series({
        "balls": len(d),
        "share%": 0.0,
        "rpo": d.team_runs.mean() * 6,
        "dot%": (d.team_runs == 0).mean() * 100,
        "bdry%": (d.is_four | d.is_six).mean() * 100,
        "wkt%": d.is_wicket.mean() * 100,
        "|wpa|": d.wpa.abs().mean(),
    })


lg = b[b.legal]
for k in ["pA", "pB", "pC", "pD"]:
    t = lg.groupby(k).apply(summ).round(2)
    t["share%"] = (t.balls / t.balls.sum() * 100).round(1)
    print(f"\n== {k}  (True = high pressure)")
    print(t)
    print("phase mix of flagged balls:", lg[lg[k]].phase.value_counts(normalize=True).round(2).to_dict(),
          "| innings mix:", lg[lg[k]].innings.value_counts(normalize=True).round(2).to_dict())

# adjust for phase: compare within phase so the table is not just 'more death overs'
print("\n== pD within phase x innings")
for (inn, ph), d in lg.groupby(["innings", "phase"]):
    hi, lo = d[d.pD], d[~d.pD]
    if len(hi) > 200:
        print(inn, ph, f"hi n={len(hi)} rpo={hi.team_runs.mean()*6:.2f} dot={100*(hi.team_runs==0).mean():.1f} "
              f"wkt={100*hi.is_wicket.mean():.1f} | lo rpo={lo.team_runs.mean()*6:.2f} "
              f"dot={100*(lo.team_runs==0).mean():.1f} wkt={100*lo.is_wicket.mean():.1f}")

b["high_pressure"] = b.pD
b.to_csv(os.path.join(OUT, "state_final.csv.gz"), index=False)
print("\nsaved state_final.csv.gz; q75 =", round(q75, 5))
