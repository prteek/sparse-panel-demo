---
title: sparse-panel-modelling decisions
updated: 2026-10-27
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
- **Tuning by sequential time-series CV, 15-day look-ahead** (2026-10-09) — replaces both the earlier 3-month-ahead CV and the random-block gap-filling CV (`sparse_panel.interp_cv`, kept but unused): splits must be sequential, never random. 12 expanding folds, cutoffs every 30 days from 2016-03-20 to 2017-02-13, each validated on the next 15 days. Chosen: roughness 1e5, 30-day spacing, 100 steps — the edge of the grid (strongest smoothing, fewest steps); not widened, by choice.
- **Every station hidden, not one station per area** (2026-10-09) — a station hold-out look-ahead CV (one station per area hidden for 15 days, neighbours observed; 24 folds) was tried and deleted. Scored on hidden readings it picks no penalty (λ=0, 15-day spacing, 750 steps: sparse 0.743 vs 0.812 for λ=1e5) — closer to the full daily means but far from a monthly level (31-day MA error 0.34 vs 0.17). It rewards tracking shared 2–3 week episodes; hiding every station rewards only what persists 15 days ahead, which is what *prevailing* level is meant to be.
- **Look-ahead CV ranks settings like the truth does** — rank correlation 0.996 between the sparse-reading score and the error vs true level; chosen 0.251 vs best 0.250. (The old random-block CV was close to blind: 0.725–0.769 across settings.)
- **Joint vs single-target under these settings** — single-target is slightly better wherever a pollutant has its own data (gap blocks PM10 0.135 vs 0.144, PM2.5 0.148 vs 0.161); joint wins only on a fully dropped series (post: 0.144 vs 0.182). The post was rewritten around this.
- **Downward bias from logging few-hour means** (checked 2026-10-08) — joint model sat −0.13 log below (−0.11 with the current settings) the true level for both pollutants; the textbook lognormal shift (+½σ²) overshoots to +0.14. Not corrected yet.

# Demo and repo

- **Precompute, don't fit live** — the app only reads tables, so it is instant and reproducible on a free CPU Space; one network fit takes ~3 s, all 9 take ~30 s offline.
- **Thinning is the only operational setting** — global Poisson mean hours per station-day, λ ∈ {0.5, 1.2, 3}; 1.2 is the blog post's setting.
- **Station labels are the view** — the viewer picks a group, not a station, and sees every station in it. Labels: regular (4), sparse (2, λ × 0.4), very sparse (2, 15% of days kept), long gap (2, 90–120 day gaps instead of 20–60), dropped series (2, one pollutant removed).
- **Labels balanced by area** — each degraded label gets one urban and one suburban station (only 4 suburban stations exist); assigned once with seed 0.
- **Common random numbers across λ** — hour order, hour-count uniforms, day-keep draws and gap positions are fixed per cell, and hour counts use the inverse Poisson CDF, so a higher λ keeps a superset of a lower λ's hours and scenarios differ only by λ.
- **Network settings from the blog post's CV** — roughness 1e5, bump spacing 30 days, 100 steps (look-ahead CV, see Method); not re-tuned per scenario.
- **±15-day mean falls back to the area** — where a station has no readings within ±15 days, the same window over its area's readings is used (as in the blog post).
- **Dropped series scored on the dropped pollutant only** — the other pollutant at those stations keeps its data, so including it would flatter every estimator.
- **Dropped label ignores the pollutant control** — the two dropped stations lost different pollutants (Dongsi PM2.5, Dingling PM10), so each panel shows its own dropped series.
- **Predictions published as a separate HF dataset** — the app downloads `prteek/sparse-panel-demo` unless local `outputs/` exist; the data is derived from UCI data, so it carries CC BY 4.0.
- **Gradio on an HF Space** — switched from Streamlit: HF no longer offers a Streamlit SDK (only gradio/docker/static), so Gradio runs natively without a Dockerfile; Gradio/Docker Spaces on cpu-basic need PRO.
- **2026-10-26: Write-up moved in; one shared library** — regression_room is being split into one repo per post. The Beijing write-up and its scripts moved in and import `sparse_panel`, the same package the Space, dataset build and model card use. The post's `prep.py`, `cv.py`, `interp_cv.py` and `evaluate.py` became `sparse_panel.data` (blog post section), `.cv`, `.interp_cv` and `.evaluate`; the post's `model.py` was already identical to `sparse_panel.model`, and its truth/±15-day mean were already `sparse_panel.baselines`.
- **Both thinnings kept** — the blog post's uniform thinning (`thin_uniform`) and the demo's label-based one (`thin_by_label`) answer different questions, and the post's numbers depend on the former's exact random sequence, so neither replaces the other.
- **Station-file order pinned** — the original code stacked the station CSVs in glob (filesystem) order before a seeded shuffle; alphabetical order draws a different sample (23,218 vs 23,243 readings). `BLOG_STATION_ORDER` records the original order; `tests/test_blog_data.py` checks the data. Verified: regenerated data identical to the post's files, and every number in the post reproduced.
- **CV results tracked in `results/`** — the blog post reads `cv_results.csv` to pick its settings and the CV takes long to run, so the results are versioned; the demo's `outputs/` and fitted weights stay out of git.
- **`smooth_truth` keeps the `n = 1` column** — the CV scoring weights by `n`; precompute drops it so the published dataset schema is unchanged (re-run verified identical tables).
- **2026-10-27: Blog post** — the write-up is a Hugging Face community article (huggingface.co/blog/prteek/sparse-panel-modelling) rendered from `blog/index.qmd` to markdown. Figures are saved to `blog/assets/` and linked locally; each is uploaded in the article editor, which replaces its link. The post renders from the single `index.qmd` (no Quarto project); its notes live in the root summary, decisions and todolist. The post's data code is named for it (`BLOG_LAMBDA`, `prepare_blog_data`, `tests/test_blog_data.py`).
- **2026-10-27: No `huggingface/` folder** — `blog/`, `space/`, `dataset/`, `model/`, `publish.sh` and `wait_for_space.py` sit at the repo root; the workflows and `publish.sh` use the new paths, and the Space's local fallback looks for `outputs/` one level up.
