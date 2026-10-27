---
title: sparse-panel-modelling summary
updated: 2026-10-27
---

- **App is live** (Gradio): huggingface.co/spaces/prteek/sparse-panel-demo, cpu-basic; all 5 groups × 3 thinning levels × 2 pollutants checked locally, and the live Space checked through its API.
- **Precompute**: 9 network fits in ~30 s; outputs 3.6 MB (`predictions.parquet` 3.1 MB).
- **At the blog post's thinning (λ = 1.2)**, mean absolute log error vs the 31-day MA, joint / single-target / ±15-day mean: regular 0.152 / 0.145 / 0.162, sparse 0.182 / 0.179 / 0.243, very sparse 0.191 / 0.187 / 0.460, long gap 0.151 / 0.143 / 0.171, dropped series 0.173 / 0.190 / 0.148.
- **Dropped series differs from the blog post**: the ±15-day mean (area fallback) beats the joint network (0.148 vs 0.173); in the blog post the joint model won (0.144 vs 0.156). Not yet investigated.
- **Published**: code at github.com/prteek/sparse-panel-modelling, data at huggingface.co/datasets/prteek/sparse-panel-demo (both public).
- **GAM**: not included; in the backlog.
- **Blog post** (`blog/index.qmd`): renders to markdown for huggingface.co/blog/prteek/sparse-panel-modelling, figures to `blog/assets/`.
- **Blog data:** 12 stations × 1,461 days; sparse version has both pollutants on 44% of days, one on 45%, 1.7 hours per observed day. Full data is the ground truth.
- **Blog tuning: sequential 15-day look-ahead CV** (`sparse_panel.cv`, `results/cv_summary.md`): 12 expanding folds over the last year; picks the strongest smoothing (roughness 1e5, 30-day spacing, 100 steps), at the edge of the grid. Sparse-reading score tracks error vs true level almost perfectly (rank corr 0.996); chosen 0.251 vs trailing 30-day mean 0.27–0.35. A one-station-per-area hold-out CV was tried and dropped: it picks a much rougher, episode-tracking curve (see decisions.md).
- **Single-target slightly beats joint where a pollutant has its own data** (`results/evaluation.md`): test gaps PM10 0.135 vs 0.144, PM2.5 0.148 vs 0.161; ±15-day mean 0.38–0.40.
- **Models denoise heavily:** raw sparse readings miss the true level by 0.84 (PM2.5) / 0.70 (PM10); the networks by 0.14–0.17.
- **Blog experiment:** long gaps joint 0.156 / single 0.144 / ±15-day 0.292; very sparse station 0.152 / 0.141 / 0.496; dropped series joint 0.144 vs single 0.182 vs area average 0.156 (only case where joint wins); dense days ±15-day 0.153 vs joint 0.154.
- **Open issues:** the model sits ~11% below the true level (log-of-few-hours bias); heavy smoothing flattens short episodes; best setting is at the grid edge.
