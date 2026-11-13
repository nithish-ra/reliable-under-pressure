"""Step 1: replay every innings ball by ball and attach the match state *before* each delivery.

Output: out/state.parquet-like CSV (out/state.csv.gz) with one row per delivery (innings 1-2 only).
Run:  python analysis/01_build_state.py
"""
import os
import numpy as np
import pandas as pd

DATA = r"C:\Users\sivar\OneDrive\Desktop\RR\data\Data Pack"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

b = pd.read_csv(os.path.join(DATA, "ball_by_ball.csv"), low_memory=False)
m = pd.read_csv(os.path.join(DATA, "matches.csv"))
c = pd.read_csv(os.path.join(DATA, "competitions.csv"))

# main innings only; super overs (3, 4) are a different game
b = b[b.innings.isin([1, 2])].copy()
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

# ---- per-ball run accounting -------------------------------------------------
extras = b.extras1_runs.fillna(0) + b.extras_2_runs.fillna(0)
b["team_runs"] = b.runs + extras
b["bat_runs"] = b.runs  # byes / leg-byes sit in extras, so this is runs off the bat
bowler_charged = b.extras_type.isin(["wide", "Overstep"])
b["bowl_runs"] = b.runs + np.where(bowler_charged, extras, 0)
b["legal"] = ~b.extras_type.isin(["wide", "Overstep"])
b["is_wicket"] = b.is_wicket.astype(bool)
# wickets credited to the bowler (not run-outs / retired / obstruction)
b["bowler_wkt"] = b.is_wicket & b.dismissal_type.isin(["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"])
b["is_four"] = b.is_four.astype(bool)
b["is_six"] = b.is_six.astype(bool)
# a ball faced by the striker (wides are not faced)
b["bat_faced"] = b.extras_type.ne("wide")

# ---- state before each ball --------------------------------------------------
g = b.groupby(["match_id", "innings"])
b["runs_before"] = g.team_runs.cumsum() - b.team_runs
b["wkts_before"] = g.is_wicket.cumsum() - b.is_wicket
b["legal_before"] = g.legal.cumsum() - b.legal

# ---- match info: overs, target, winner --------------------------------------
m = m.merge(c[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")
inn1_total = b[b.innings == 1].groupby("match_id").team_runs.sum().rename("inn1_total")
m = m.merge(inn1_total, on="match_id", how="left")
m["max_overs_1"] = m.first_inn_revised_overs.fillna(20)
m["max_overs_2"] = m.second_inn_revised_overs.fillna(20)
m["target"] = m.revised_target.fillna(m.inn1_total + 1)

keep = ["match_id", "comp_id", "comp_name", "match_format", "season", "match_date", "stage", "knock_out",
        "venue_id", "win_team_id", "result", "target", "max_overs_1", "max_overs_2", "inn1_total"]
b = b.merge(m[keep], on="match_id", how="inner")

b["max_balls"] = np.where(b.innings == 1, b.max_overs_1, b.max_overs_2) * 6
b["balls_left"] = (b.max_balls - b.legal_before).clip(lower=0)
b["wkts_in_hand"] = 10 - b.wkts_before
b["crr"] = np.where(b.legal_before > 0, b.runs_before * 6 / b.legal_before.replace(0, np.nan), np.nan)

is2 = b.innings == 2
b["runs_needed"] = np.where(is2, b.target - b.runs_before, np.nan)
b["rrr"] = np.where(is2 & (b.balls_left > 0), b.runs_needed * 6 / b.balls_left.replace(0, np.nan), np.nan)
b["rrr_minus_crr"] = b.rrr - b.crr.fillna(0)

# ---- phase -------------------------------------------------------------------
over = (b.legal_before // 6).astype(int)  # 0-indexed over of the *next legal ball*
b["phase"] = np.select([over < 6, over < 16], ["PP", "Middle"], default="Death")

# ---- outcome label from the batting side's point of view --------------------
batting_won = (b.win_team_id == b.batting_team_id)
valid = b.win_team_id.isin([0]) | b.win_team_id.isna()
decided = (b.win_team_id == b.batting_team_id) | (b.win_team_id == b.bowling_team_id)
b["decided"] = decided
b["bat_won"] = np.where(decided, batting_won.astype(int), np.nan)

print("rows", len(b), "matches", b.match_id.nunique(), "decided rows", int(decided.sum()))
print(b.phase.value_counts(normalize=True).round(3).to_dict())
print(b.groupby("match_format").match_id.nunique().to_dict())
print(b[is2].rrr.describe().round(2).to_dict())
b.to_csv(os.path.join(OUT, "state.csv.gz"), index=False)
print("saved", os.path.join(OUT, "state.csv.gz"))
