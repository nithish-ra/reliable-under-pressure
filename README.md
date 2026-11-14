# Reliable Players Under Pressure: A Win-Probability Rating System for T20 Cricket

A data-driven system that measures how batters and bowlers perform in high-pressure match situations compared with
their usual level, ranks the most reliable performers, quantifies the uncertainty of every rank, and links the results
to a franchise's squad needs. It was built for the SupeRR Selector Season 2 (Round 1) problem, using the Rajasthan
Royals data pack: ball-by-ball IPL 2022-26 and T20I 2022-26 records, 842 completed matches and 199,029 deliveries.

The write-up is in `SupeRR_Round1_Report.pdf`. This file documents the system itself.

> **Data note.** The data pack is proprietary and was provided for the competition only. This project contains code and
> derived results; the raw pack and the ball-level intermediate tables are not part of what is shared.

**Links.** Live dashboard: https://nithish-ra.github.io/reliable-under-pressure/ · Code and method: https://github.com/nithish-ra/reliable-under-pressure · Video walkthrough: https://drive.google.com/file/d/1qqTnCi7Rh9NWhAinxVIhbp0PeOLMPN8w/view?usp=sharing

---

## 1. Result

Top 10 batters and bowlers under pressure (ranked by the final rating; confidence is the share of 400 match-level
bootstrap resamples in which the player stays in the top 10):

| # | Batter | Confidence | Bowler | Confidence |
|---|---|---|---|---|
| 1 | Tim David | 85% | Sunil Narine | 94% |
| 2 | Suryakumar Yadav | 87% | Sufiyan Muqeem | 82% |
| 3 | Harry Brook | 77% | Jasprit Bumrah | 80% |
| 4 | Heinrich Klaasen | 86% | Kuldeep Yadav | 74% |
| 5 | Jos Buttler | 58% | Rashid Khan | 67% |
| 6 | Shreyas Iyer | 57% | Abrar Ahmed | 56% |
| 7 | Andre Russell | 53% | Mujeeb Ur Rahman | 52% |
| 8 | Rajat Patidar | 53% | Mohammad Nawaz | 49% |
| 9 | Cameron Green | 44% | Varun Chakravarthy | 47% |
| 10 | Dinesh Karthik | 54% | Krunal Pandya | 33% |

The top four on each list are firm (74-94%). Ranks 5-10 are close contests (33-67%); the next-best players (Kohli and
Shashank Singh among batters; Pathirana, Hasaranga and Hazlewood among bowlers) are close behind.

The ranking is purely performance-based. The organisers confirmed that all players in the dataset are eligible regardless of
nationality, capped status, franchise or auction availability, so no one is excluded. For practical reference only, if Karthik
(reported retired) and the two Pakistani bowlers, Abrar Ahmed and Mohammad Nawaz, were unobtainable, two replacement rules are provided:

1. **Next in the overall ranking:** Virat Kohli (batters); Matheesha Pathirana and Wanindu Hasaranga (bowlers).
2. **Like-for-like (same role, IPL-proven):** Kohli and Shashank Singh for Karthik; Hasaranga and Noor Ahmad for the two spinners.

---

## 2. Design highlights

1. **Pressure defined by stakes.** A calibrated win-probability model gives every ball a *leverage*: the expected
   win-probability swing of that delivery. High pressure = high leverage, game still open, and a chase / death-over /
   collapse context.
2. **One engine for batting and bowling.** Both sides are scored from the same ball-level value, mirror-imaged.
3. **Usual level is built in.** Each player's pressure record is blended with his own ordinary-ball baseline.
4. **Metric chosen by repeatability.** Candidate metrics were compared by split-half repeatability and out-of-sample
   prediction. The first metric (win-probability added) scored 0.03 / 0.06 and was replaced; the final blend scores 0.39 / 0.37.
5. **Uncertainty reported for every rank.** Bootstrap over whole matches gives P(top 10) and rank ranges; players are
   grouped into Core / Probable / Bubble tiers.
6. **Out-of-time backtest.** Players are rated on matches before a cut-off date and tested on pressure balls after it.
7. **Fair-comparison controls.** Expected values are set per innings x phase x match tier (IPL / T20I vs full members /
   T20I vs associates / A-team events) x pressure flag; minimum samples; resampling by match rather than by ball.
8. **Reliability archetypes.** Players are classed as elite and steady, elite and rising, or rising from a modest base.
9. **Evidence-quality flags.** The share of each player's pressure balls against associate or A-team opposition.
10. **Team-level diagnostics from the same engine.** Win probability gained and lost per match by phase, for every
    franchise, reconciled to actual results.
11. **Availability layer (supplementary, not used in ranking).** Each ranked player additionally carries an availability
    flag, like-for-like same-role alternates, and a next-in-ranking fallback order if flagged players are unavailable.

---

## 3. Pipeline

All scripts are in `analysis/` and run in numeric order; intermediate tables are written to `analysis/out/` (created when the scripts run).

| Step | Script | Purpose | Key output |
|---|---|---|---|
| 1 | `01_build_state.py` | Replays every innings ball by ball; adds score, wickets, balls left, target (DLS-aware), CRR, RRR, phase and the winner label | `state.csv.gz` (200,961 balls) |
| 2 | `02_pressure_model.py` | Win-probability model (gradient-boosted trees, one per innings, 5-fold CV grouped by match); per-ball leverage and win-probability added | AUC 0.886 (chase), 0.668 (first innings); well calibrated |
| 3 | `03_define_pressure.py` | Compares four pressure definitions and fixes the final flag | 17,937 flagged balls (9.4%) |
| 4 | `04_ratings.py` | First-draft ratings using win-probability added (kept for the audit trail) | metric found to be noisy |
| 5 | `05_robustness.py` | Stability checks and split-half test of the draft metric | repeatability 0.03-0.07 |
| 6 | `06_metric_selection.py` | Compares five candidate metrics; estimates the run-cost of a wicket | blend with own baseline wins |
| 7 | `07_final_ranking.py` | Tier-adjusted ball value; blend tuned on held-out matches; bootstrap; robustness variants | `final_*.csv`, `tuning_grid.csv`, `split_half_final.csv` |
| 8 | `08_names_and_rr.py` | Player-name matching; first franchise profile | name / id checks |
| 9 | `09_rr_analytics.py` | Results, phase profile, win-probability points by team x phase, squad profile | `rr_*.csv` |
| 10 | `10_charts.py` | All figures (colour palette validated for colour-vision safety) | `fig/*.png` |
| 12 | `12_backtest.py` | Walk-forward backtest | `backtest.csv` |

### Report and dashboard

`docs/` holds the finished report (`SupeRR_Round1_Report.pdf`) and the interactive dashboard (`docs/index.html`, a single self-contained static page served by GitHub Pages). Both are built from the result tables produced by the steps above and show aggregated results only; no ball-by-ball data is included.

---

## 4. Method

**Match state.** Before each delivery: runs, wickets in hand, balls left, target (revised target under DLS), current
and required run rate. Super overs are excluded. For the match clock, wides and no-balls do not use a ball; for player
values every delivery counts, wides and no-balls are charged to the bowler as runs, wides are not faced by batters, and
byes and leg-byes are not charged to the bowler.

**Win probability and leverage.** Win probability is predicted from the state (separate models for the first innings
and the chase). Leverage is the expected absolute change in win probability over the league's outcome mix for that
innings and phase (dot, 1, 2, 3, 4, 6, wicket).

**Pressure flag.** A ball is high-pressure when all hold:
(i) leverage is in the top 25%; (ii) the batting side's win probability is between 20% and 80%; (iii) it is a chase
ball, a first-innings death-over ball, or a first-innings ball after 4+ wickets (powerplay balls only after 3+ wickets).
High-pressure balls move win probability about 2.8x more than ordinary balls, with a higher wicket rate (+22%) and a
lower dot-ball rate (-14%).

**Ball value (runs units).**
Batter: `v = (runs - E[runs]) - 4.7 x (dismissal - E[dismissal])`. Bowler: the mirror image.
`E[.]` is the league mean for the same innings x phase x match tier x pressure flag. The 4.7 runs-per-wicket exchange
rate comes from regressing win-probability change on runs and wickets over pressure balls (a run is worth +2.5
win-probability points, a wicket -11.7). Only wickets credited to the bowler count for bowlers.

**Rating.** `rating = (sum of v on pressure balls + 400 x baseline) / (pressure balls + 400)`, where `baseline` is the
player's own mean v on ordinary balls, shrunk toward zero with 300 pseudo-balls. A player with 400 pressure balls is
weighted 50/50 between pressure record and baseline; one with 100 balls, 20/80. The weight of 400 was selected on
held-out matches: held-out predictive correlation rises steeply to about 400 and then flattens (0.151 at 400 vs 0.157
at the best finite setting), while keeping about 30% weight on the pressure record for a typical player.

**Eligibility.** At least 60 pressure balls, 10 pressure matches and 100 ordinary balls: 91 batters and 102 bowlers.

**Uncertainty.** Bootstrap over whole matches (400 resamples) gives P(top 10) and the 10th-90th percentile rank.

**Archetypes.** *Elite* = ordinary-ball level in the top quarter of rated players. *Rising / slipping* = pressure level
more than 15 runs per 100 balls above / below the player's own ordinary level.

**Team accounting.** A team's win-probability points per match, summed over batting and bowling, equal its win rate
minus 50 on average; the franchise-level tables use this identity as a consistency check (Rajasthan Royals: +3.1 points,
about 53% expected vs 51.3% actual over 76 IPL matches).

---

## 5. Validation

| Check | Result |
|---|---|
| Win model discrimination | AUC 0.886 chase, 0.668 first innings (grouped CV) |
| Win model calibration | predicted vs actual win rate align across deciles |
| Pressure balls behave differently | win-prob swing x2.8; wicket rate +22%; dot rate -14% |
| Split-half repeatability | win-prob added 0.03 / 0.06; pressure record only 0.08 / 0.07; **final blend 0.39 / 0.37** |
| Blend strength tuned on unseen matches | best finite 0.157; chosen 0.151 |
| Sensitivity (blend weight, looser / tighter flag, no associates, IPL-only, T20I-only) | top four stable on both lists; ranks 5-10 move |
| Walk-forward backtest | bowlers: blend predicts later pressure performance (+0.25, +0.16) and beats the pressure-only record; batters mixed (-0.02, +0.17) |
| Team accounting | RR net +3.1 pts/match ~ 53% expected vs 51.3% actual |

---

## 6. Findings

- **Most of "reliable under pressure" is "good overall".** Ordinary-ball level predicts later pressure performance
  almost as well as any other signal; pressure-specific skill is too small to detect with about 100 pressure balls per player.
- **Two kinds of reliability.** Elite and steady (Tim David, Suryakumar Yadav, Klaasen, Patidar; Narine, Bumrah, Rashid
  Khan, Abrar, Nawaz, Varun) and elite and rising (Brook, Buttler, Iyer, Russell, Green; Muqeem, Kuldeep, Mujeeb, Krunal).
  Karthik rises from a modest base and is discounted toward it.
- **Rajasthan Royals' profile (IPL 2022-26).** Powerplay batting and bowling both rank 1st of 10 franchises. The leaks are
  middle-overs batting (#8), death-overs batting (#8), middle-overs bowling (#7) and death-overs bowling (#9, the largest,
  concentrated in the second innings). Closing the four gaps to league average is worth about 1.2 wins per 14 games;
  to the top-three average, about 2.9 (accounting arithmetic, not a forecast).
- **Squad ratings.** In the IPL 2026 squad, Jaiswal rates 15th of 91 batters; the death-overs pace group rates 84th-98th of 102 bowlers.

---

## 7. Development log: issues found and corrected

1. The first ranking metric (win-probability added) had near-zero split-half repeatability and was replaced.
2. Uncapped empirical-Bayes shrinkage diverged for bowlers (prior strength about 7.7 million balls); it was capped, then replaced by the tuned blend.
3. All "T20" matches had been treated as IPL; the pack also contains two A-team / associate tournaments, which became a separate tier, and everything was re-run.
4. A robustness run overwrote the tuning grid with a variant's grid; outputs are now written per variant.
5. Ball accounting and the median pressure-ball count were stated incorrectly in a draft and corrected.
6. A backtest statement ("pressure-only record always weakest") held in 3 of 4 tests and was corrected.
7. Second-innings splits for bowlers had been labelled "chase" / "defend" inverted; they now read "defending a total" / "bowling first".

## 8. Limitations

- Small samples: the median rated player has about 100 pressure balls.
- No detectable clutch-specific skill beyond general quality.
- First-innings pressure is less certain because the first-innings model is weaker.
- Opposition and batting position are controlled only through phase and match tier.
- Some players' evidence is mostly T20I or associate-level (Muqeem 30% against associate / A-team sides).
- The batter backtest rests on 40-46 players per test and is mixed.
- Franchise diagnostics rest on 76 IPL matches for Rajasthan Royals.
- Nationality, retirement and eligibility flags are supplementary, play no role in the rankings, and come from public knowledge, not from the data pack, which holds no contracts, purses or retentions.

---

## 9. Reproducing the results

Requirements: Python 3.13; `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`.

```
python analysis/01_build_state.py
python analysis/02_pressure_model.py
python analysis/03_define_pressure.py
python analysis/04_ratings.py
python analysis/05_robustness.py             # draft-metric stability and split-half test (prints to console)
python analysis/06_metric_selection.py       # candidate-metric comparison and wicket run-cost (prints to console)
python analysis/07_final_ranking.py          # blend weight chosen by the held-out grid; writes tuning_grid.csv and split_half_final.csv
# report setting (blend weight 400) and sensitivity variants, selected through environment variables:
#   KB=400 KZ=300 TAG=_k400            (also KB=150 TAG=_k150, KB=800 TAG=_k800)
#   VARIANT=q65 | q85 | noassoc | iplonly | t20ionly   with KB=400 KZ=300 TAG=_v_<variant>
python analysis/08_names_and_rr.py           # name matching and first franchise profile (prints to console)
python analysis/09_rr_analytics.py
python analysis/12_backtest.py
python analysis/10_charts.py
```

Random seeds are fixed.

## 10. Layout

```
README.md                      this file
docs/                          finished report (PDF) and interactive dashboard (index.html)
analysis/                      pipeline scripts 01-10 and 12 (the data pack and all derived tables are not included)
```
