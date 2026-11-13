"""Step 4: player ratings for batters and bowlers under pressure.

Per player (batting: balls faced, bowling: balls bowled):
  wpa_p     mean win-probability added per ball on high-pressure balls (bowler = -WPA); empirical-Bayes shrunk
  delta     pressure run-index minus ordinary run-index (index = actual / league-expected runs for the same
            innings x phase cell; for bowlers inverted so >1 is good)  -> 'vs usual level'
  consist   share of matches (with >= 4 pressure balls) in which the player's pressure WPA was positive, shrunk
  composite 0.70*z(wpa_p) + 0.15*z(delta) + 0.15*z(consist)
Input: out/state_final.csv.gz   Output: out/batters.csv, out/bowlers.csv
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
b = pd.read_csv(os.path.join(OUT, "state_final.csv.gz"), low_memory=False)
b["is_ipl"] = b.comp_name.str.startswith("IPL")  # the pack also holds two A-team / associate T20 tournaments

MIN_P_BALLS = {"bat": 60, "bowl": 60}
MIN_P_MATCHES = 10
BOOT = 400
K_CAP = 600
rng = np.random.default_rng(11)


def side_frame(role, data):
    if role == "bat":
        d = data[data.bat_faced].copy()
        d["pid"], d["pname"] = d.striker_id, d.striker_name
        d["val"] = d.wpa
        d["runs"] = d.bat_runs
    else:
        d = data.copy()
        d["pid"], d["pname"] = d.bowler_id, d.bowler_name
        d["val"] = -d.wpa
        d["runs"] = d.bowl_runs
    d["cell"] = d.innings.astype(str) + "_" + d.phase
    exp = d.groupby("cell").runs.transform("mean")  # league-expected runs / ball in this innings x phase
    d["exp_runs"] = exp
    return d


def shrink_k(g, col="val"):
    """Method-of-moments empirical-Bayes prior strength k = sigma_w^2 / tau^2 (in balls)."""
    s = g.groupby("pid")[col].agg(["mean", "count", "var"])
    s = s[s["count"] >= 30]
    within = (s["var"] * s["count"]).sum() / s["count"].sum()
    tau2 = max(s["mean"].var() - (within / s["count"]).mean(), 1e-9)
    return within / tau2, within


def rate(role, data=None, flag="high_pressure", save=True, min_balls=None, min_matches=None, verbose=True):
    data = b if data is None else data
    min_balls = MIN_P_BALLS[role] if min_balls is None else min_balls
    min_matches = MIN_P_MATCHES if min_matches is None else min_matches
    d = side_frame(role, data)
    d["high_pressure"] = d[flag]
    names = d.groupby("pid").pname.agg(lambda x: x.value_counts().index[0])
    hp = d[d.high_pressure]
    k_est, within = shrink_k(hp)
    k = min(k_est, K_CAP)  # method-of-moments tau^2 ~ 0 for bowlers => uncapped k explodes; cap documented in report
    if verbose:
        print(f"[{role}] EB prior strength estimated k = {k_est:.0f}, used k = {k:.0f} balls (within-var {within:.5f})")

    # per-match pressure sums
    pm = hp.groupby(["pid", "match_id"]).agg(w=("val", "sum"), n=("val", "size")).reset_index()
    pl = hp.groupby("pid").agg(p_balls=("val", "size"), p_sum=("val", "sum"), p_runs=("runs", "sum"),
                               p_exp=("exp_runs", "sum"), p_matches=("match_id", "nunique"))
    pl["wpa_raw"] = pl.p_sum / pl.p_balls
    pl["wpa_p"] = pl.p_sum / (pl.p_balls + k)  # shrink towards league mean (~0)
    pl["p_index"] = pl.p_runs / pl.p_exp
    od = d[~d.high_pressure].groupby("pid").agg(o_balls=("val", "size"), o_runs=("runs", "sum"),
                                                o_exp=("exp_runs", "sum"))
    od["o_index"] = od.o_runs / od.o_exp
    pl = pl.join(od, how="left")
    sign = 1 if role == "bat" else -1  # bowler: lower runs conceded is better
    pl["delta"] = sign * (pl.p_index - pl.o_index)

    # consistency over matches with >= 4 pressure balls
    mm = pm[pm.n >= 4]
    cs = mm.groupby("pid").w.agg(pos=lambda x: (x > 0).sum(), m="size")
    pl = pl.join(cs, how="left")
    pl["consist"] = (pl.pos.fillna(0) + 2.5) / (pl.m.fillna(0) + 5)  # shrunk toward 0.5

    # IPL / T20I split (raw mean WPA)
    for tag, sel in [("ipl", hp.is_ipl), ("t20i", ~hp.is_ipl)]:
        t = hp[sel].groupby("pid").val.agg(["sum", "size"])
        pl[f"wpa_{tag}"] = (t["sum"] / t["size"]).reindex(pl.index)
        pl[f"n_{tag}"] = t["size"].reindex(pl.index)

    el = pl[(pl.p_balls >= min_balls) & (pl.p_matches >= min_matches)].copy()

    # bootstrap over matches -> 10th percentile of mean WPA per ball
    lo = {}
    grp = {p: g for p, g in pm.groupby("pid") if p in el.index}
    for p, g in grp.items():
        w, n = g.w.values, g.n.values
        idx = rng.integers(0, len(g), size=(BOOT, len(g)))
        lo[p] = np.percentile(w[idx].sum(1) / n[idx].sum(1), 10)
    el["wpa_p10"] = pd.Series(lo)

    z = lambda s: (s - s.mean()) / s.std()
    el["composite"] = 0.70 * z(el.wpa_p) + 0.15 * z(el.delta.clip(-1, 1)) + 0.15 * z(el.consist)
    el["name"] = names.reindex(el.index)
    el = el.sort_values("composite", ascending=False)
    el["rank"] = np.arange(1, len(el) + 1)
    cols = ["rank", "name", "p_balls", "p_matches", "wpa_raw", "wpa_p", "wpa_p10", "p_index", "o_index", "delta",
            "consist", "composite", "wpa_ipl", "n_ipl", "wpa_t20i", "n_t20i"]
    el = el[cols]
    if save:
        el.to_csv(os.path.join(OUT, f"{'batters' if role == 'bat' else 'bowlers'}.csv"))
    if verbose:
        print(f"[{role}] eligible players: {len(el)}")
    return el


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for role in ("bat", "bowl"):
        el = rate(role)
        print(el.head(20).round(4).to_string())
        print()
