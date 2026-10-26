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
