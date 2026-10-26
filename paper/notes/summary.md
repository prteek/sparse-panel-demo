---
title: Summary
updated: 2026-10-09
---

- **Data:** 12 stations × 1,461 days; sparse version has both pollutants on 44% of days, one on 45%, 1.7 hours per observed day ([decisions.md](./decisions.md#data)). Full data is the ground truth.
- **Tuning: sequential 15-day look-ahead CV** (`cv.py`, `outputs/cv_summary.md`): 12 expanding folds over the last year; picks the strongest smoothing (roughness 1e5, 30-day spacing, 100 steps), at the edge of the grid. Sparse-reading score tracks error vs true level almost perfectly (rank corr 0.996); chosen 0.251 vs trailing 30-day mean 0.27–0.35. A one-station-per-area hold-out CV was tried and dropped: it picks a much rougher, episode-tracking curve (see decisions.md).
- **Single-target slightly beats joint where a pollutant has its own data** (`outputs/evaluation.md`): test gaps PM10 0.135 vs 0.144, PM2.5 0.148 vs 0.161; ±15-day mean 0.38–0.40. All estimators improved vs the old settings (joint was 0.166/0.168).
- **Models denoise heavily:** raw sparse readings miss the true level by 0.84 (PM2.5) / 0.70 (PM10); the networks by 0.14–0.17.
- **Post experiment** (`index.qmd`): long gaps joint 0.156 / single 0.144 / ±15-day 0.292; very sparse station 0.152 / 0.141 / 0.496; dropped series joint 0.144 vs single 0.182 vs area average 0.156 (only case where joint wins); dense days ±15-day 0.153 vs joint 0.154.
- **Open issues:** the model sits ~11% below the true level (log-of-few-hours bias); heavy smoothing flattens short episodes; best setting is at the grid edge.
