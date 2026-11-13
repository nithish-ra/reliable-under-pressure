"""Step 8a: (1) check the shortlisted names against the mapping file (dropdown matching),
(2) profile Rajasthan Royals' own pressure performance by phase to find squad gaps,
(3) player role descriptors for the squad-fit notes.
Run: python analysis/08_names_and_rr.py
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = r"C:\Users\sivar\OneDrive\Desktop\RR\data\Data Pack"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

mp = pd.read_csv(os.path.join(DATA, "competition_team_player_mapping.csv"))
teams = pd.read_csv(os.path.join(DATA, "teams.csv"))
comps = pd.read_csv(os.path.join(DATA, "competitions.csv"))
bat = pd.read_csv(os.path.join(OUT, "final_batters_k400.csv")).head(16)
bowl = pd.read_csv(os.path.join(OUT, "final_bowlers_k400.csv")).head(16)

# ---- 1. names vs mapping --------------------------------------------------------
dup = mp.groupby("player_name").player_id.nunique()
print("names mapped to >1 player id:", dup[dup > 1].to_dict())
for lbl, df in (("BAT", bat), ("BOWL", bowl)):
    rows = []
    for _, r in df.iterrows():
        mm = mp[mp.player_id == r.pid]
        nm = sorted(mm.player_name.unique())
        sameName_ids = mp[mp.player_name.isin(nm)].player_id.unique()
        last_ipl = mm[mm.comp_name.str.startswith("IPL")].sort_values("comp_name").tail(1)
        rows.append({"rank": r["rank"], "pid": r.pid, "ball_name": r["name"], "mapping_names": "|".join(nm),
                     "n_ids_for_name": len(sameName_ids),
                     "last_ipl": (last_ipl.comp_name.iloc[0] + " / " + last_ipl.team_name.iloc[0]) if len(last_ipl) else "-"})
    print(f"\n{lbl}"); print(pd.DataFrame(rows).to_string(index=False))

# ---- 2. RR own pressure profile ---------------------------------------------------
rr_ids = teams[teams.team_name.str.contains("Rajasthan", case=False)].team_id.unique()
print("\nRR team ids:", rr_ids)
rr26 = mp[mp.comp_name.eq("IPL 2026") & mp.team_id.isin(rr_ids)]
print("RR IPL 2026 squad in mapping:", len(rr26)); print(sorted(rr26.player_name.tolist()))

b = pd.read_csv(os.path.join(OUT, "state_final.csv.gz"), low_memory=False)
b = b[b.comp_name.str.startswith("IPL")]  # IPL
b["cell"] = b.innings.astype(str) + "_" + b.phase
hp = b[b.high_pressure]
for side, col, sgn in (("RR batting", "batting_team_id", 1), ("RR bowling", "bowling_team_id", -1)):
    rows = []
    for ph in ("PP", "Middle", "Death"):
        for inn in (1, 2):
            d = hp[(hp.phase == ph) & (hp.innings == inn)]
            if len(d) < 50:
                continue
            r = d[d[col].isin(rr_ids)]
            if len(r) < 30:
                continue
            rows.append({"phase": ph, "innings": inn, "rr_balls": len(r),
                         "rr_rpo": r.team_runs.mean() * 6, "lg_rpo": d.team_runs.mean() * 6,
                         "rr_wkt%": r.is_wicket.mean() * 100, "lg_wkt%": d.is_wicket.mean() * 100,
                         "rr_dot%": (r.team_runs == 0).mean() * 100, "lg_dot%": (d.team_runs == 0).mean() * 100})
    print(f"\n{side}, IPL 2022-26, high-pressure balls (batting: higher rpo/lower wkt good; bowling: opposite)")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

# overall by season
print("\nRR pressure balls per season, batting rpo vs league / bowling rpo vs league")
for s, d in hp.groupby("season"):
    rb, rw = d[d.batting_team_id.isin(rr_ids)], d[d.bowling_team_id.isin(rr_ids)]
    print(s, f"bat {rb.team_runs.mean()*6:.2f} (lg {d.team_runs.mean()*6:.2f}) wkt% {rb.is_wicket.mean()*100:.1f} | "
             f"bowl {rw.team_runs.mean()*6:.2f} wkt% {rw.is_wicket.mean()*100:.1f}  n={len(rb)}/{len(rw)}")

# ---- 3. role descriptors ---------------------------------------------------------------
full = pd.read_csv(os.path.join(DATA, "ball_by_ball.csv"), usecols=["striker_id", "batting_type", "batting_order", "bowler_id", "bowling_type", "pace_or_spin"], low_memory=False)
print("\nBATTER roles")
for _, r in bat.iterrows():
    f = full[full.striker_id == r.pid]
    print(f"{r['rank']:>2} {r['name']:<20} {f.batting_type.mode().iloc[0]} | median order {f.batting_order.median():.0f} | "
          f"death share of pressure balls: {r.n_death/r.pn:.0%} | chase v {r.v_chase:.2f} / bat1st v {r.v_bat1st:.2f} | "
          f"mid v {r.v_mid:.2f} / death v {r.v_death:.2f} | srl {r.p_sr:.0f}")
print("\nBOWLER roles")
for _, r in bowl.iterrows():
    f = full[full.bowler_id == r.pid]
    print(f"{r['rank']:>2} {r['name']:<20} {f.bowling_type.mode().iloc[0]} ({f.pace_or_spin.mode().iloc[0]}) | "
          f"mid v {r.v_mid:.2f} (n={r.n_mid:.0f}) / death v {r.v_death:.2f} (n={r.n_death:.0f}) | "
          f"chase {r.v_chase:.2f} / bat1st {r.v_bat1st:.2f} | econ {r.p_econ:.2f}")
