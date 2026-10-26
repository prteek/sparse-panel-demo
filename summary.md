---
title: sparse-panel-demo summary
updated: 2026-10-26
---

- **App works locally**: all 5 groups × 3 thinning levels × 2 pollutants render without errors (Streamlit AppTest), checked visually in light/dark-safe colours.
- **Precompute**: 9 network fits in ~30 s; outputs 3.6 MB (`predictions.parquet` 3.1 MB).
- **At the post's thinning (λ = 1.2)**, mean absolute log error vs the 31-day MA, joint / single-target / ±15-day mean: regular 0.152 / 0.145 / 0.162, sparse 0.182 / 0.179 / 0.243, very sparse 0.191 / 0.187 / 0.460, long gap 0.151 / 0.143 / 0.171, dropped series 0.173 / 0.190 / 0.148.
- **Dropped series differs from the post**: the ±15-day mean (area fallback) beats the joint network (0.148 vs 0.173); in the post the joint model won (0.144 vs 0.156). Not yet investigated.
- **Code is on GitHub**: github.com/prteek/sparse-panel-demo (public). HF dataset and Space are pending (see `todolist.txt`).
- **GAM**: not included; in the backlog.
