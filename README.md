# sparse-panel-demo

An interactive Gradio demo of the [Beijing air quality post](https://prteek.github.io/regression_room/posts/beijing_air_quality/) (paper on [alphaXiv](https://www.alphaxiv.org/abs/2610.pm25-pm10-joint-smoothing)): hourly PM2.5 and PM10 from 12 stations are thinned to a few readings a day, with gaps, and three estimators try to recover the slow underlying level (the 31-day moving average of the full data). Stations are grouped by what was done to them (regular, sparse, very sparse, long gap, dropped series); the viewer picks a global thinning level and a group, and sees every station in that group. All fits are precomputed offline; the app only reads the results.

**Live app:** https://huggingface.co/spaces/prteek/sparse-panel-demo · **Data:** https://huggingface.co/datasets/prteek/sparse-panel-demo

- `app.py` — Gradio app; reads `outputs/` (or `DATA_DIR`) if present, otherwise the HF dataset `prteek/sparse-panel-demo`
- `precompute.py` — fits every scenario (joint network, single-target networks, ±15-day mean) and writes `outputs/`
- `sparse_panel/data.py` — UCI download, daily means, station labels and label-dependent thinning
- `sparse_panel/model.py` — the joint two-target network, copied from the post
- `sparse_panel/baselines.py` — 31-day MA benchmark and the ±15-day mean with area fallback
- `hf/` — HF dataset card, Space config (`space/`: README front matter, requirements) and `publish.sh` to upload both
- `outputs/` — precomputed tables (git-ignored; published as the HF dataset)
- `data/` — raw UCI data (git-ignored; downloaded by `precompute.py`)
- `summary.md` — current state
- `decisions.md` — choices and why
- `todolist.txt` — remaining work, including the GAM backlog

```bash
uv sync
uv run python precompute.py     # ~30 s: 3 thinning levels x 3 network fits
uv run python app.py
```

Data: Beijing Multi-Site Air Quality, UCI Machine Learning Repository, CC BY 4.0.
