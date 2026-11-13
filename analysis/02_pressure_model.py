"""Step 2: win-probability model -> per-ball leverage (pressure) and win-probability-added (WPA).

Pressure definition: LEVERAGE = expected |change in win probability| from the next ball, given the current state.
High pressure = top quartile of leverage. All predictions are out-of-fold (grouped by match) so no leakage.

Input : out/state.csv.gz      Output: out/state_wp.csv.gz
Run   : python analysis/02_pressure_model.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
b = pd.read_csv(os.path.join(OUT, "state.csv.gz"), low_memory=False)
b = b[b.decided].reset_index(drop=True)
RUNS = np.array([0, 1, 2, 3, 4, 6])


def feats(inn, runs, wk_hand, bl, maxb, tgt):
    """Feature matrix for a (possibly hypothetical) state. inn is 1 or 2 (scalar)."""
    played = np.maximum(maxb - bl, 1)
    crr = runs * 6 / played
    if inn == 1:
        return np.column_stack([runs, wk_hand, bl, crr])
    need = tgt - runs
    rrr = np.clip(np.where(bl > 0, need * 6 / np.maximum(bl, 1), 36), -6, 36)
    return np.column_stack([need, wk_hand, bl, rrr, crr, rrr - crr])


def wp(model, inn, runs, wk_hand, bl, maxb, tgt):
    p = model.predict_proba(feats(inn, runs, wk_hand, bl, maxb, tgt))[:, 1]
    if inn == 2:  # exact terminal states in the chase
        need = tgt - runs
        p = np.where(need <= 0, 1.0, np.where((wk_hand <= 0) | (bl <= 0), 0.0, p))
    return p


# outcome distribution of a legal ball by innings x phase (league-wide)
legal = b[b.legal]
outc = {}
for (inn, ph), d in legal.groupby(["innings", "phase"]):
    pw = d.is_wicket.mean()
    r = d[~d.is_wicket].bat_runs.replace({5: 4}).clip(upper=6)
    rp = np.array([(r == k).mean() for k in RUNS])
    outc[(inn, ph)] = (pw, rp / rp.sum())

b["wp_before"] = np.nan
b["wp_after"] = np.nan
b["leverage"] = np.nan

gkf = GroupKFold(n_splits=5)
oof = {}
for inn in (1, 2):
    idx = b.index[b.innings == inn]
    d = b.loc[idx]
    X = feats(inn, d.runs_before.values, d.wkts_in_hand.values, d.balls_left.values,
              d.max_balls.values, d.target.values)
    y = d.bat_won.astype(int).values
    pred = np.zeros(len(d))
    for tr, te in gkf.split(X, y, groups=d.match_id.values):
        mdl = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=250,
                                             l2_regularization=1.0, min_samples_leaf=80, random_state=7)
        mdl.fit(X[tr], y[tr])
        te_idx = idx[te]
        t = b.loc[te_idx]
        args = (inn, t.runs_before.values, t.wkts_in_hand.values, t.balls_left.values,
                t.max_balls.values, t.target.values)
        p0 = wp(mdl, *args)
        # actual next state
        runs1 = t.runs_before.values + t.team_runs.values
        wk1 = t.wkts_in_hand.values - t.is_wicket.values.astype(int)
        bl1 = np.maximum(t.balls_left.values - t.legal.values.astype(int), 0)
        p1 = wp(mdl, inn, runs1, wk1, bl1, t.max_balls.values, t.target.values)
        # expected |dWP| over hypothetical outcomes -> leverage
        lev = np.zeros(len(t))
        ph = t.phase.values
        for key in set(zip([inn] * len(t), ph)):
            msk = ph == key[1]
            pw, rp = outc[key]
            tot = pw * np.abs(wp(mdl, inn, t.runs_before.values[msk], t.wkts_in_hand.values[msk] - 1,
                                 np.maximum(t.balls_left.values[msk] - 1, 0), t.max_balls.values[msk],
                                 t.target.values[msk]) - p0[msk])
            for k, r in enumerate(RUNS):
                tot += (1 - pw) * rp[k] * np.abs(wp(mdl, inn, t.runs_before.values[msk] + r,
                                                    t.wkts_in_hand.values[msk],
                                                    np.maximum(t.balls_left.values[msk] - 1, 0),
                                                    t.max_balls.values[msk], t.target.values[msk]) - p0[msk])
            lev[msk] = tot
        b.loc[te_idx, "wp_before"] = p0
        b.loc[te_idx, "wp_after"] = p1
        b.loc[te_idx, "leverage"] = lev
    m_ = b.loc[idx]
    print(f"innings {inn}: OOF AUC={roc_auc_score(y, m_.wp_before):.3f} "
          f"logloss={log_loss(y, m_.wp_before.clip(1e-4, 1-1e-4)):.3f} brier={brier_score_loss(y, m_.wp_before):.3f}")

b["wpa"] = b.wp_after - b.wp_before  # +ve = good for batting side, -ve credit to the bowler

# ---- pressure threshold and diagnostics --------------------------------------
lg = b[b.legal].copy()
q75 = lg.leverage.quantile(0.75)
q90 = lg.leverage.quantile(0.90)
b["high_pressure"] = b.leverage >= q75
lg["high_pressure"] = lg.leverage >= q75
print(f"\nleverage: median={lg.leverage.median():.4f} q75={q75:.4f} q90={q90:.4f}")
print("share of high-pressure balls by phase x innings:")
print(lg.groupby(["innings", "phase"]).high_pressure.mean().round(3).unstack())


def summ(d):
    return pd.Series({
        "balls": len(d),
        "runs_per_over": d.team_runs.mean() * 6,
        "dot%": (d.team_runs == 0).mean() * 100,
        "boundary%": (d.is_four | d.is_six).mean() * 100,
        "wkt%": d.is_wicket.mean() * 100,
        "|WPA|/ball": d.wpa.abs().mean(),
    })


print("\nHigh pressure vs ordinary (legal balls):")
print(lg.groupby("high_pressure").apply(summ).round(2))
print("\nBy innings x phase:")
print(lg.groupby(["innings", "phase", "high_pressure"]).apply(summ).round(2))

# calibration of the win-probability model
lg["bin"] = pd.cut(lg.wp_before, np.linspace(0, 1, 11))
print("\ncalibration (predicted bin -> actual bat win rate):")
print(lg.groupby("bin", observed=True).bat_won.agg(["mean", "size"]).round(3))

b.to_csv(os.path.join(OUT, "state_wp.csv.gz"), index=False)
print("saved state_wp.csv.gz  threshold(q75) =", round(q75, 5))
