---
title: sparse-panel-demo decisions
updated: 2026-10-26
---

- **Precompute, don't fit live** — the app only reads tables, so it is instant and reproducible on a free CPU Space; one network fit takes ~3 s, all 9 take ~30 s offline.
- **Thinning is the only operational setting** — global Poisson mean hours per station-day, λ ∈ {0.5, 1.2, 3}; 1.2 is the paper's setting.
- **Station labels are the view** — the viewer picks a group, not a station, and sees every station in it. Labels: regular (4), sparse (2, λ × 0.4), very sparse (2, 15% of days kept), long gap (2, 90–120 day gaps instead of 20–60), dropped series (2, one pollutant removed).
- **Labels balanced by area** — each degraded label gets one urban and one suburban station (only 4 suburban stations exist); assigned once with seed 0.
- **Common random numbers across λ** — hour order, hour-count uniforms, day-keep draws and gap positions are fixed per cell, and hour counts use the inverse Poisson CDF, so a higher λ keeps a superset of a lower λ's hours and scenarios differ only by λ.
- **Network settings from the paper's CV** — roughness 1e5, bump spacing 30 days, 100 steps (look-ahead CV in the paper); not re-tuned per scenario.
- **±15-day mean falls back to the area** — where a station has no readings within ±15 days, the same window over its area's readings is used (as in the paper).
- **Dropped series scored on the dropped pollutant only** — the other pollutant at those stations keeps its data, so including it would flatter every estimator.
- **Dropped label ignores the pollutant control** — the two dropped stations lost different pollutants (Dongsi PM2.5, Dingling PM10), so each panel shows its own dropped series.
- **Predictions published as a separate HF dataset** — the app downloads `prteek/sparse-panel-demo` unless local `outputs/` exist; the data is derived from UCI data, so it carries CC BY 4.0.
- **Gradio on an HF Space** — switched from Streamlit: HF no longer offers a Streamlit SDK (only gradio/docker/static), so Gradio runs natively without a Dockerfile; Gradio/Docker Spaces on cpu-basic need PRO.
- **2026-10-26: Paper moved in; one shared library** — regression_room is being split into one repo per post. The Beijing write-up now lives in `paper/` (prose copied verbatim; only the setup cell's imports changed) and imports `sparse_panel`, the same package the Space, dataset build and model card use. The post's `prep.py`, `cv.py`, `interp_cv.py` and `evaluate.py` became `sparse_panel.data` (paper section), `.cv`, `.interp_cv` and `.evaluate`; the post's `model.py` was already identical to `sparse_panel.model`, and its truth/±15-day mean were already `sparse_panel.baselines`.
- **Both thinnings kept** — the paper's uniform thinning (`thin_uniform`) and the demo's label-based one (`thin_by_label`) answer different questions, and the paper's numbers depend on the former's exact random sequence, so neither replaces the other.
- **Paper's station-file order pinned** — the original code stacked the station CSVs in glob (filesystem) order before a seeded shuffle; alphabetical order draws a different sample (23,218 vs 23,243 readings). `PAPER_STATION_ORDER` records the original order; `tests/test_paper_data.py` checks the data. Verified: regenerated data identical to the post's files, and all 74 numbers and 2,122 words of the re-rendered PDF identical to the published one.
- **CV results tracked in `results/`** — the paper reads `cv_results.csv` to pick its settings and the CV takes long to run, so the results are versioned; the demo's `outputs/` and fitted weights stay out of git.
- **`smooth_truth` keeps the paper's `n = 1` column** — the CV scoring weights by `n`; precompute drops it so the published dataset schema is unchanged (re-run verified identical tables).
