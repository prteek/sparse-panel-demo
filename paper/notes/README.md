# Beijing air quality: joint smooth model of PM2.5 and PM10

Tests the "one network, two related targets, smooth level, missing target masked" approach on public data with a known ground truth: hourly readings from 12 Beijing monitoring stations (UCI Beijing Multi-Site Air Quality, 2013-03 to 2017-02), thinned to be as sparse and noisy as a typical small-area panel.

- `index.qmd` — the blog post; runs the experiment (dropped series, very sparse station, three estimators) and plots. Render from the repo root with the repo `.venv` active (or `QUARTO_PYTHON=.venv/bin/python`). `thumbnail.png` is a hand-picked image, not generated.
- `README.md`, `summary.md`, `decisions.md`, `todolist.txt` — project docs (not rendered: `_quarto.yml` renders only `.qmd`).
- `prep.py` — builds `data/daily_full.parquet` (ground truth), `data/daily_sparse.parquet` (thinned) and `data/gaps.parquet`.
- `model.py` — encoding, network, loss, roughness penalty, training loop.
- `cv.py` — look-ahead time series CV: 12 sequential expanding folds, every station hidden for the 15 days after each cutoff; `outputs/cv_summary.md`, `cv_results.csv`. Used for tuning.
- `interp_cv.py` — earlier gap-filling CV with random hidden 30-day blocks; `outputs/interp_cv_summary.md`. Kept for reference, not used (splits must not be random).
- `evaluate.py` — final fits (joint, single-target, ±15-day mean) with the `cv.py` settings vs ground truth by day type; `outputs/evaluation.md`, `predictions.parquet`.
- `plots.py` — `outputs/series.png`, `outputs/gap_zoom.png`.
- `outputs/` — results.
- `data/` — raw station CSVs (UCI) and prepared parquet files.
