"""Step 9: Rajasthan Royals diagnostics (IPL 2022-2026).
Outputs CSVs in out/: rr_seasons.csv, rr_phase.csv, rr_wpa_team.csv, rr_squad.csv, rr_pressure_bowlers.csv, rr_pressure_batters.csv
Run: python analysis/09_rr_analytics.py
"""
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = r"C:\Users\sivar\OneDrive\Desktop\RR\data\Data Pack"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
RR = 3

m = pd.read_csv(os.path.join(DATA, "matches.csv")).merge(
    pd.read_csv(os.path.join(DATA, "competitions.csv"))[["comp_id", "comp_name", "match_format", "season"]], on="comp_id")
mp = pd.read_csv(os.path.join(DATA, "competition_team_player_mapping.csv"))
b = pd.read_csv(os.path.join(OUT, "state_final.csv.gz"), low_memory=False)
b = b[b.comp_name.str.startswith("IPL")]  # IPL only
ipl = m[m.comp_name.str.startswith("IPL")].copy()
first_bat = b[b.innings == 1].groupby("match_id").batting_team_id.first().rename("first_bat")
ipl = ipl.merge(first_bat, on="match_id", how="left")


def margin(r):
    x = re.search(r"Won by (\d+) (run|wicket)", str(r))
    return (int(x.group(1)), x.group(2)) if x else (np.nan, None)


ipl[["margin", "mtype"]] = ipl.result.apply(lambda r: pd.Series(margin(r)))
ipl["close"] = ((ipl.mtype == "run") & (ipl.margin <= 10)) | ((ipl.mtype == "wicket") & (ipl.margin <= 3))

# ---- 1. RR results ------------------------------------------------------------------
r = ipl[(ipl.home_team_id == RR) | (ipl.away_team_id == RR)].copy()
r = r[r.win_team_id.notna() & (r.win_team_id > 0)]  # decided
r["win"] = r.win_team_id == RR
r["chasing"] = r.first_bat != RR


def tab(d):
    return pd.Series({"M": len(d), "W": int(d.win.sum()), "win%": 100 * d.win.mean(),
                      "chase_M": int(d.chasing.sum()), "chase_W": int((d.win & d.chasing).sum()),
                      "bat1_M": int((~d.chasing).sum()), "bat1_W": int((d.win & ~d.chasing).sum()),
                      "close_M": int(d.close.sum()), "close_W": int((d.win & d.close).sum()),
                      "ko_M": int(d.knock_out.sum()), "ko_W": int((d.win & (d.knock_out == 1)).sum())})


seasons = r.groupby("season").apply(tab)
seasons.loc["ALL"] = tab(r)
print("RR results:\n", seasons.round(1).to_string())
seasons.to_csv(os.path.join(OUT, "rr_seasons.csv"))

# league comparators: win% batting first vs chasing across all IPL games
al = ipl[ipl.win_team_id.notna() & (ipl.win_team_id > 0)].copy()
al["bat1_win"] = al.win_team_id == al.first_bat
print("\nleague bat-first win%:", round(100 * al.bat1_win.mean(), 1), "| close games share:", round(100 * al.close.mean(), 1))

# ---- 2. phase profile (all balls) ---------------------------------------------------
lg = b[b.legal]
rows = []
for side, col in (("bat", "batting_team_id"), ("bowl", "bowling_team_id")):
    for ph in ("PP", "Middle", "Death"):
        d = lg[lg.phase == ph]
        rr = d[d[col] == RR]
        rows.append({"side": side, "phase": ph, "rr_balls": len(rr), "rr_rpo": rr.team_runs.mean() * 6,
                     "lg_rpo": d.team_runs.mean() * 6, "rr_wkt100": rr.is_wicket.mean() * 100,
                     "lg_wkt100": d.is_wicket.mean() * 100, "rr_bdry%": (rr.is_four | rr.is_six).mean() * 100,
                     "lg_bdry%": (d.is_four | d.is_six).mean() * 100, "rr_dot%": (rr.team_runs == 0).mean() * 100,
                     "lg_dot%": (d.team_runs == 0).mean() * 100})
ph = pd.DataFrame(rows)
print("\nPhase profile, all balls (IPL 2022-26):\n", ph.round(2).to_string(index=False))
ph.to_csv(os.path.join(OUT, "rr_phase.csv"), index=False)

# ---- 3. win-probability generated per match, by team x phase x side --------------------
dec = b[b.decided & b.wpa.notna()]
res = []
nm = dec.groupby("batting_team_id").match_id.nunique()
for side, col, sgn in (("bat", "batting_team_id", 1), ("bowl", "bowling_team_id", -1)):
    g = dec.assign(w=sgn * dec.wpa).groupby([col, "phase"]).w.sum().unstack()
    matches = dec.groupby(col).match_id.nunique()
    g = g.div(matches, axis=0) * 100  # win-prob % points per match
    g["side"] = side
    res.append(g.reset_index().rename(columns={col: "team_id"}))
w = pd.concat(res)
names = pd.read_csv(os.path.join(DATA, "teams.csv")).drop_duplicates("team_id").set_index("team_id").team_name
w["team"] = w.team_id.map(names)
w = w[w.team_id.isin(dec.batting_team_id.unique()) & (dec.groupby("batting_team_id").match_id.nunique().reindex(w.team_id).values >= 40)]
print("\nWin-probability points generated per match (IPL 2022-26), by team:")
for side in ("bat", "bowl"):
    t = w[w.side == side].set_index("team")[["PP", "Middle", "Death"]]
    t["total"] = t.sum(1)
    t["RR_rank_total"] = t.total.rank(ascending=False).astype(int)
    for c in ("PP", "Middle", "Death"):
        t[f"rk_{c}"] = t[c].rank(ascending=False).astype(int)
    print(f"-- {side}\n", t.sort_values("total", ascending=False).round(2).to_string())
w.to_csv(os.path.join(OUT, "rr_wpa_team.csv"), index=False)

# by innings, RR only
rrw = []
for side, col, sgn in (("bat", "batting_team_id", 1), ("bowl", "bowling_team_id", -1)):
    d = dec[dec[col] == RR]
    n = d.match_id.nunique()
    rrw.append(d.assign(w=sgn * d.wpa).groupby(["innings", "phase"]).w.sum().div(n).mul(100).rename(side))
print("\nRR WPA points per match by innings x phase:\n", pd.concat(rrw, axis=1).round(2).to_string())

# ---- 4. RR squad & recent pressure usage -----------------------------------------------
sq = mp[mp.comp_name.eq("IPL 2026") & (mp.team_id == RR)][["player_id", "player_name"]]
bb = pd.read_csv(os.path.join(DATA, "ball_by_ball.csv"), usecols=["striker_id", "batting_type", "bowler_id", "bowling_type", "pace_or_spin"], low_memory=False)
rows = []
fb, fw = pd.read_csv(os.path.join(OUT, "final_batters_k400.csv")), pd.read_csv(os.path.join(OUT, "final_bowlers_k400.csv"))
for _, p in sq.iterrows():
    nb, nw = (bb.striker_id == p.player_id).sum(), (bb.bowler_id == p.player_id).sum()
    ht = bb[bb.striker_id == p.player_id].batting_type.mode()
    bt = bb[bb.bowler_id == p.player_id]
    rows.append({"pid": p.player_id, "name": p.player_name, "balls_faced": nb, "balls_bowled": nw,
                 "bat_hand": ht.iloc[0] if len(ht) else "", "bowl_type": bt.bowling_type.mode().iloc[0] if nw > 60 else "",
                 "pace_spin": bt.pace_or_spin.mode().iloc[0] if nw > 60 else "",
                 "bat_rank": fb.set_index("pid")["rank"].get(p.player_id, np.nan),
                 "bowl_rank": fw.set_index("pid")["rank"].get(p.player_id, np.nan)})
squad = pd.DataFrame(rows).sort_values("balls_faced", ascending=False)
print("\nRR 2026 squad profile:\n", squad.to_string(index=False))
squad.to_csv(os.path.join(OUT, "rr_squad.csv"), index=False)

rec = b[(b.season.astype(str).isin(["2025", "2026"])) & b.high_pressure]
pb = rec[rec.bowling_team_id == RR].groupby("bowler_name").agg(
    pressure_balls=("legal", "sum"), runs=("bowl_runs", "sum"), wkts=("bowler_wkt", "sum"),
    dots=("team_runs", lambda x: (x == 0).mean() * 100), death_share=("phase", lambda x: (x == "Death").mean() * 100))
pb["econ"] = pb.runs / pb.pressure_balls * 6
print("\nRR bowlers on high-pressure balls, IPL 2025-26:\n", pb.sort_values("pressure_balls", ascending=False).head(10).round(2).to_string())
pb.to_csv(os.path.join(OUT, "rr_pressure_bowlers.csv"))
pt = rec[(rec.batting_team_id == RR) & rec.bat_faced].groupby("striker_name").agg(
    pressure_balls=("bat_runs", "size"), runs=("bat_runs", "sum"), outs=("is_wicket", "sum"))
pt["sr"] = pt.runs / pt.pressure_balls * 100
print("\nRR batters on high-pressure balls, IPL 2025-26:\n", pt.sort_values("pressure_balls", ascending=False).head(10).round(1).to_string())
pt.to_csv(os.path.join(OUT, "rr_pressure_batters.csv"))
