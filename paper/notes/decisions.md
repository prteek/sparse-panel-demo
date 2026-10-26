---
title: Decisions
updated: 2026-10-09
---

# Data

- **Daily station level** — hourly data is 98% complete; daily keeps realistic noise (daily log PM2.5 sd 0.90 vs 0.32 for monthly) while weekly or monthly would hide all missingness.
- **Deliberate thinning** — each station-day keeps a Poisson(1.2) random subset of valid hours per pollutant (independently), plus 2 gaps of 20–60 days per station and pollutant: 44% of days with both observed, 45% with one, 1.7 hours per observed day. The full data is the ground truth.
- **Raw values kept** — 999 caps and hours with PM2.5 > PM10 (5%) are left in as real-world noise.
- **Area grouping** — 8 urban-core and 4 suburban stations; station embedding is an offset from its area.
- **True level** — 31-day centred mean of log full-data daily means. Note the ±15-day baseline uses the same window shape, which favours it on densely observed days.

# Method

- **Same principles as the price prototype** — two heads (log PM2.5, log PM10), masked missing target, Gaussian NLL with variance σ²/n + τ² (n = hours), smooth time bumps, roughness penalty on every station's full series, small station offsets.
- **Small init for area embeddings** (std 0.1) so they are actually learned within a short training run.
- **Error metric** — hour-weighted mean absolute log error; MAPE is unstable for values from 2 to 999.
- **Tuning by sequential time-series CV, 15-day look-ahead** (2026-10-09) — replaces both the earlier 3-month-ahead CV and the random-block gap-filling CV (`interp_cv.py`, kept but unused): splits must be sequential, never random. 12 expanding folds, cutoffs every 30 days from 2016-03-20 to 2017-02-13, each validated on the next 15 days. Chosen: roughness 1e5, 30-day spacing, 100 steps — the edge of the grid (strongest smoothing, fewest steps); not widened, by choice.
- **Every station hidden, not one station per area** (2026-10-09) — a station hold-out look-ahead CV (one station per area hidden for 15 days, neighbours observed; 24 folds) was tried and deleted. Scored on hidden readings it picks no penalty (λ=0, 15-day spacing, 750 steps: sparse 0.743 vs 0.812 for λ=1e5) — closer to the full daily means but far from a monthly level (31-day MA error 0.34 vs 0.17). It rewards tracking shared 2–3 week episodes; hiding every station rewards only what persists 15 days ahead, which is what *prevailing* level is meant to be.
- **Look-ahead CV ranks settings like the truth does** — rank correlation 0.996 between the sparse-reading score and the error vs true level; chosen 0.251 vs best 0.250. (The old random-block CV was close to blind: 0.725–0.769 across settings.)
- **Joint vs single-target under these settings** — single-target is slightly better wherever a pollutant has its own data (gap blocks PM10 0.135 vs 0.144, PM2.5 0.148 vs 0.161); joint wins only on a fully dropped series (post: 0.144 vs 0.182). The post was rewritten around this.
- **Downward bias from logging few-hour means** (checked 2026-10-08) — joint model sat −0.13 log below (−0.11 with the current settings) the true level for both pollutants; the textbook lognormal shift (+½σ²) overshoots to +0.14. Not corrected yet.
