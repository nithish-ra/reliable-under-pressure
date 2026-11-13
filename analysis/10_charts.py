"""Step 10: all figures for the report -> out/fig/*.png  (palette validated: blue/pink/orange + neutrals)"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, DBLUE, LBLUE, PINK, ORANGE, GRAY = "#2a78d6", "#1c5cab", "#86b6ef", "#e5197e", "#eb6834", "#9a9992"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def clean(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------- data ----------------
b = pd.read_csv(os.path.join(OUT, "state_final.csv.gz"), low_memory=False)
lg = b[b.legal]
fb = pd.read_csv(os.path.join(OUT, "final_batters_k400.csv"))
fw = pd.read_csv(os.path.join(OUT, "final_bowlers_k400.csv"))
grid = pd.read_csv(os.path.join(OUT, "tuning_grid.csv"))

# ---------------- fig1: pressure definition ----------------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
bins = np.linspace(0, 1, 11)
for inn, col, lab in ((1, GRAY, "1st innings"), (2, BLUE, "Chase")):
    d = lg[(lg.innings == inn)]
    g = d.groupby(pd.cut(d.wp_before, bins), observed=True).bat_won.mean()
    ax[0].plot((bins[:-1] + bins[1:])[: len(g)] / 2 + 0.05 - 0.05, g.values, color=col, lw=1.8, marker="o", ms=3)
ax[0].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
ax[0].text(0.52, 0.12, "Chase", color=BLUE, fontweight="bold", fontsize=8)
ax[0].text(0.05, 0.62, "1st innings", color=GRAY, fontweight="bold", fontsize=8)
ax[0].set_xlabel("Model win probability")
ax[0].set_ylabel("Actual win rate")
ax[0].set_title("Win model is calibrated")
clean(ax[0], "both")

share = lg.groupby(["phase", "innings"]).high_pressure.mean().unstack() * 100
order = ["PP", "Middle", "Death"]
x = np.arange(3)
ax[1].bar(x - 0.2, share.loc[order, 1], 0.38, color=GRAY, label="1st innings")
ax[1].bar(x + 0.2, share.loc[order, 2], 0.38, color=BLUE, label="Chase")
for i, o in enumerate(order):
    ax[1].text(i - 0.2, share.loc[o, 1] + 0.6, f"{share.loc[o,1]:.0f}%", ha="center", fontsize=7, color=INK2)
    ax[1].text(i + 0.2, share.loc[o, 2] + 0.6, f"{share.loc[o,2]:.0f}%", ha="center", fontsize=7, color=INK2)
ax[1].set_xticks(x, ["Powerplay", "Middle", "Death"])
ax[1].set_ylabel("% of balls flagged")
ax[1].set_title("Where pressure occurs")
ax[1].set_ylim(0, 27)
ax[1].text(-0.45, 24.5, "1st innings", color=GRAY, fontweight="bold")
ax[1].text(-0.45, 22.3, "Chase", color=BLUE, fontweight="bold")
clean(ax[1])

hp, od = lg[lg.high_pressure], lg[~lg.high_pressure]
idx = {"Win-prob swing\nper ball": hp.wpa.abs().mean() / od.wpa.abs().mean(),
       "Wicket rate": hp.is_wicket.mean() / od.is_wicket.mean(),
       "Runs per over": hp.team_runs.mean() / od.team_runs.mean(),
       "Dot-ball rate": (hp.team_runs == 0).mean() / (od.team_runs == 0).mean()}
labs = list(idx)
vals = [v * 100 for v in idx.values()]
ax[2].barh(range(4), vals, color=[PINK, BLUE, BLUE, BLUE], height=0.55)
ax[2].axvline(100, color=INK2, lw=0.8, ls=":")
for i, v in enumerate(vals):
    ax[2].text(max(v, 100) + 6, i, f"{v:.0f}", va="center", fontsize=8, color=INK)
ax[2].set_yticks(range(4), labs)
ax[2].invert_yaxis()
ax[2].set_xlim(0, max(vals) * 1.18)
ax[2].set_xlabel("High-pressure balls, ordinary = 100")
ax[2].set_title("Pressure balls play differently")
clean(ax[2], "x")
fig.tight_layout(w_pad=1.2)
save(fig, "fig1_pressure.png")

# ---------------- fig2: reliability & tuning ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.05, 1]})
sh = pd.read_csv(os.path.join(OUT, "split_half_final.csv")).set_index(["role", "k_base"]).split_half
# win-prob-added numbers come from the step-06 metric comparison (0.031 / 0.064); the other two are the final metric
rel = pd.DataFrame({"Win prob.\nadded": [0.031, 0.064],
                    "Pressure balls only\n(runs + dismissals)": [sh[("bat", 0)], sh[("bowl", 0)]],
                    "Final rating\n(blend with own level)": [sh[("bat", 400)], sh[("bowl", 400)]]}, index=["Batters", "Bowlers"])
x = np.arange(3)
ax[0].bar(x - 0.2, rel.loc["Batters"], 0.38, color=BLUE, label="Batters")
ax[0].bar(x + 0.2, rel.loc["Bowlers"], 0.38, color=ORANGE, label="Bowlers")
for i in range(3):
    ax[0].text(i - 0.2, rel.iloc[0, i] + 0.008, f"{rel.iloc[0,i]:.2f}", ha="center", fontsize=7, color=INK2)
    ax[0].text(i + 0.2, rel.iloc[1, i] + 0.008, f"{rel.iloc[1,i]:.2f}", ha="center", fontsize=7, color=INK2)
ax[0].set_xticks(x, rel.columns, fontsize=6.8)
ax[0].set_ylabel("Split-half rank correlation")
ax[0].set_title("Only the blended rating is repeatable")
ax[0].set_ylim(0, 0.46)
ax[0].text(-0.4, 0.40, "Batters", color=BLUE, fontweight="bold")
ax[0].text(-0.4, 0.37, "Bowlers", color=ORANGE, fontweight="bold")
clean(ax[0])

g = grid[grid.k_zero == 300].groupby("k_base").heldout_corr.mean()
labs = ["0", "25", "50", "100", "200", "400", "800", "1600", "Baseline\nonly"]
ax[1].plot(range(len(g)), g.values, color=BLUE, lw=1.8, marker="o", ms=4)
ax[1].scatter([5], [g.values[5]], s=60, color=PINK, zorder=5)
ax[1].annotate("chosen (400):\nkeeps 30% weight\non pressure record", (5, g.values[5]), (1.0, 0.145),
               fontsize=6.8, color=PINK, arrowprops=dict(arrowstyle="-", color=PINK, lw=0.7))
ax[1].set_xticks(range(len(g)), labs, fontsize=6.5)
ax[1].set_xlabel("Weight on player's own baseline (pseudo-balls)")
ax[1].set_ylabel("Held-out predictive correlation")
ax[1].set_title("Tuned on unseen matches")
clean(ax[1])
fig.tight_layout(w_pad=1.5)
save(fig, "fig2_reliability.png")


# ---------------- fig3: dumbbells ----------------
def dumbbell(df, name, role):
    d = df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    y = np.arange(len(d))
    base, pv = d.baseline_v * 100, d.pressure_v * 100
    ax.hlines(y, base, pv, color=GRID, lw=3, zorder=1)
    ax.scatter(base, y, s=34, color=GRAY, zorder=3, label="Own ordinary-ball level")
    ax.scatter(pv, y, s=44, color=PINK, zorder=4, label="Under pressure")
    ax.set_yticks(y, [f"{int(r)}. {n}" for r, n in zip(d["rank"], d["name"])])
    for yi, p in zip(y, d.p_top10):
        ax.text(1.01, yi, f"{p*100:.0f}%", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=INK)
    ax.text(1.01, len(d) - 0.35, "P(top 10)", transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, fontweight="bold")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Runs above league-average per 100 balls (wickets costed at 4.7 runs)")
    ax.legend(loc="upper right", frameon=False, fontsize=7.5, bbox_to_anchor=(1.0, 1.0))
    ax.set_xlim(min(-2, base.min() - 3, pv.min() - 3), max(base.max(), pv.max()) * 1.35)
    ax.set_title(f"{role}: pressure level vs. each player's own normal level")
    clean(ax, "x")
    save(fig, name)


dumbbell(fb, "fig3_bat_dumbbell.png", "Batters")
dumbbell(fw, "fig3_bowl_dumbbell.png", "Bowlers")


# ---------------- fig4: rank intervals ----------------
def intervals(df, name, role):
    d = df.head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(d))
    cols = [DBLUE if p >= 0.70 else (LBLUE if p >= 0.45 else GRAY) for p in d.p_top10]
    ax.axvspan(0.5, 10.5, color="#f3f2ee", zorder=0)
    for yi, lo, hi, c in zip(y, d.rank_p10, d.rank_p90, cols):
        ax.hlines(yi, lo, hi, color=c, lw=3, zorder=2)
    ax.scatter(d["rank"], y, s=40, color=cols, zorder=3, edgecolor="white", lw=1)
    ax.set_yticks(y, [f"{int(r)}. {n}" for r, n in zip(d["rank"], d["name"])])
    ax.set_xlim(0.5, max(d.rank_p90.max() + 3, 20))
    ax.set_xlabel("Rank (dot = point estimate; bar = 10th to 90th percentile over 400 match-resamples)")
    ax.set_ylim(-1.1, len(d) - 0.4)
    ax.text(5.5, -0.8, "top-10 zone", ha="center", fontsize=7, color=INK2)
    ax.set_title(f"{role}: how firm is each position?")
    for lbl, c, yy in (("Core  P>=70%", DBLUE, 0.95), ("Probable  45-70%", LBLUE, 0.88), ("Bubble  <45%", GRAY, 0.81)):
        ax.text(0.98, yy, lbl, transform=ax.transAxes, ha="right", color=c, fontweight="bold", fontsize=8)
    clean(ax, "x")
    save(fig, name)


intervals(fb, "fig4_bat_intervals.png", "Batters")
intervals(fw, "fig4_bowl_intervals.png", "Bowlers")

# combined, compact version (top 12 of each list side by side) used in the report
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.7))
for a, df, ttl in ((axs[0], fb, "Batters"), (axs[1], fw, "Bowlers")):
    d = df.head(12).iloc[::-1]
    y = np.arange(len(d))
    cols = [DBLUE if p >= 0.70 else (LBLUE if p >= 0.45 else GRAY) for p in d.p_top10]
    a.axvspan(0.5, 10.5, color="#f3f2ee", zorder=0)
    for yi, lo, hi, c in zip(y, d.rank_p10, d.rank_p90, cols):
        a.hlines(yi, lo, hi, color=c, lw=3, zorder=2)
    a.scatter(d["rank"], y, s=34, color=cols, zorder=3, edgecolor="white", lw=1)
    a.set_yticks(y, [f"{int(r)}. {n}" for r, n in zip(d["rank"], d["name"])], fontsize=6.8)
    a.set_xlim(0.5, 50)
    a.set_ylim(-1.0, len(d) - 0.4)
    a.text(5.5, -0.75, "top-10 zone", ha="center", fontsize=6.5, color=INK2)
    a.set_xlabel("Rank (dot = estimate; bar = 10th-90th percentile)", fontsize=7)
    a.set_title(f"{ttl}: how firm is each position?", fontsize=8.5)
    clean(a, "x")
axs[1].text(0.97, 0.94, "Core P>=70%", transform=axs[1].transAxes, ha="right", color=DBLUE, fontweight="bold", fontsize=7)
axs[1].text(0.97, 0.88, "Probable 45-70%", transform=axs[1].transAxes, ha="right", color=LBLUE, fontweight="bold", fontsize=7)
axs[1].text(0.97, 0.82, "Bubble <45%", transform=axs[1].transAxes, ha="right", color=GRAY, fontweight="bold", fontsize=7)
fig.tight_layout(w_pad=1.0)
save(fig, "fig4_intervals.png")

# ---------------- fig9: rise or hold steady ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5))
for a, df, ttl in ((ax[0], fb, "Batters"), (ax[1], fw, "Bowlers")):
    x, y = df.baseline_v * 100, df.pressure_v * 100
    top = df["rank"] <= 10
    a.scatter(x[~top], y[~top], s=14, color=GRAY, alpha=0.7, zorder=2)
    a.scatter(x[top], y[top], s=85, color=PINK, zorder=4)
    lim = [min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5]
    a.plot(lim, lim, color=INK2, lw=0.8, ls=":", zorder=1)
    a.axvline(df.baseline_v.quantile(0.75) * 100, color=GRID, lw=0.8)
    for _, r in df[top].iterrows():
        a.text(r.baseline_v * 100, r.pressure_v * 100, str(int(r["rank"])), ha="center", va="center", fontsize=6.3,
               color="white", fontweight="bold", zorder=5)
    a.set_xlabel("Own ordinary-ball level (runs/100 balls above avg)")
    a.set_ylabel("Under pressure")
    a.set_title(f"{ttl}: above the dotted line = rises")
    clean(a, "both")
fig.tight_layout()
save(fig, "fig9_rise_or_steady.png")

# ---------------- RR figures ----------------
w = pd.read_csv(os.path.join(OUT, "rr_wpa_team.csv"))
seasons = pd.read_csv(os.path.join(OUT, "rr_seasons.csv"), index_col=0)
RRN = "Rajasthan Royals"
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
for a, side, ttl in ((ax[0], "bat", "Batting"), (ax[1], "bowl", "Bowling")):
    t = w[w.side == side].set_index("team")[["PP", "Middle", "Death"]]
    rr, mean = t.loc[RRN], t.mean()
    x = np.arange(3)
    a.bar(x - 0.2, mean.values, 0.38, color=GRAY)
    a.bar(x + 0.2, rr.values, 0.38, color=[PINK if v >= m_ else ORANGE for v, m_ in zip(rr.values, mean.values)])
    for i, c in enumerate(["PP", "Middle", "Death"]):
        rk = int(t[c].rank(ascending=False)[RRN])
        yv = rr[c]
        a.text(i + 0.2, yv + (0.5 if yv >= 0 else -0.5), f"#{rk}", ha="center", va="bottom" if yv >= 0 else "top",
               fontsize=8, fontweight="bold", color=INK)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(x, ["Powerplay", "Middle", "Death"])
    a.set_ylim(-6.5, 14)
    a.set_title(f"{ttl}: win probability added per match")
    clean(a)
ax[0].set_ylabel("Win-prob. points per match")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=GRAY, label="League average"), Patch(color=PINK, label="RR (above average)"),
                    Patch(color=ORANGE, label="RR (below average)")], loc="lower center", ncol=3, frameon=False,
           fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
fig.text(0.5, -0.13, "#n = RR's rank among the 10 franchises (1 = best)", ha="center", fontsize=7.5, color=INK2)
fig.tight_layout()
save(fig, "fig5_rr_phase.png")

fig, ax = plt.subplots(figsize=(3.6, 2.5))
s = seasons.drop("ALL")
ax.bar(s.index.astype(str), s["win%"], color=PINK, width=0.55)
for i, (k, r) in enumerate(s.iterrows()):
    ax.text(i, r["win%"] + 1.5, f"{int(r.W)}-{int(r.M-r.W)}", ha="center", fontsize=7.5, fontweight="bold")
ax.axhline(50, color=INK2, lw=0.8, ls=":")
ax.set_ylim(0, 75)
ax.set_ylabel("Win %")
ax.set_title("RR IPL results by season")
clean(ax)
save(fig, "fig6_rr_seasons.png")

squad = pd.read_csv(os.path.join(OUT, "rr_squad.csv"))
nb, nw = len(fb), len(fw)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7), sharex=True)
for a, col, n, ttl in ((ax[0], "bat_rank", nb, "Batters"), (ax[1], "bowl_rank", nw, "Bowlers")):
    d = squad.dropna(subset=[col]).sort_values(col, ascending=False)
    pct = (1 - (d[col] - 1) / (n - 1)) * 100
    y = np.arange(len(d))
    a.hlines(y, 0, pct, color=GRID, lw=2)
    a.scatter(pct, y, s=36, color=[PINK if p >= 50 else ORANGE for p in pct], zorder=3)
    a.set_yticks(y, d["name"], fontsize=7.5)
    for yi, p, r in zip(y, pct, d[col]):
        a.text(p + 2.5, yi, f"{int(r)}/{n}", va="center", fontsize=7, color=INK2)
    a.axvline(50, color=INK2, lw=0.8, ls=":")
    a.set_xlim(0, 108)
    a.set_title(f"RR squad {ttl.lower()}: pressure rating")
    a.set_xlabel("Percentile among rated players (100 = best)")
    clean(a, "x")
fig.tight_layout()
save(fig, "fig7_rr_squad.png")

# ---- scenario: closing the gaps ----
rows = []
for side, ph in (("bat", "Middle"), ("bat", "Death"), ("bowl", "Middle"), ("bowl", "Death")):
    t = w[w.side == side].set_index("team")[ph]
    rr = t[RRN]
    rows.append({"label": f"{'Batting' if side=='bat' else 'Bowling'}: {ph.lower()} overs", "rr": rr,
                 "to_avg": max(0, t.mean() - rr), "to_top3": max(0, t.nlargest(3).mean() - rr)})
sc = pd.DataFrame(rows)
print(sc.round(2).to_string(), "\nTOTAL to avg:", sc.to_avg.sum().round(2), "to top3:", sc.to_top3.sum().round(2))
sc.to_csv(os.path.join(OUT, "rr_scenario.csv"), index=False)
fig, ax = plt.subplots(figsize=(7.2, 2.5))
y = np.arange(len(sc))[::-1]
ax.barh(y + 0.2, sc.to_top3, 0.36, color=LBLUE, label="to reach top-3 average")
ax.barh(y - 0.2, sc.to_avg, 0.36, color=PINK, label="to reach league average")
for yi, a_, t_ in zip(y, sc.to_avg, sc.to_top3):
    ax.text(a_ + 0.1, yi - 0.2, f"+{a_:.1f}", va="center", fontsize=7.5)
    ax.text(t_ + 0.1, yi + 0.2, f"+{t_:.1f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y, sc.label)
ax.set_xlabel("Win-probability points per match RR would gain")
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
ax.set_title("Size of RR's four weak spots")
clean(ax, "x")
save(fig, "fig8_scenario.png")
