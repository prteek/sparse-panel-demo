# sparse-panel-demo

Smooth levels from sparse, gappy panels, on hourly PM2.5 and PM10 from 12 Beijing stations. This repo holds
everything for the paper [PM2.5, PM10 ~ Station + Time, Analysis](https://www.alphaxiv.org/abs/2610.pm25-pm10-joint-smoothing)
on alphaXiv: the write-up, the code it runs, and the Hugging Face artefacts built from the same code.

**Live app:** https://huggingface.co/spaces/prteek/sparse-panel-demo · **Model:** https://huggingface.co/prteek/sparse-panel-demo · **Data:** https://huggingface.co/datasets/prteek/sparse-panel-demo

## Layout

```
sparse_panel/          shared library, used by everything below
  data.py              UCI download, daily means; the paper's uniform thinning and the demo's label-based thinning
  model.py             the joint two-target network (fit / predict)
  baselines.py         31-day truth, ±15-day mean with area fallback
  cv.py                the paper's look-ahead CV and best_config()      python -m sparse_panel.cv
  interp_cv.py         the paper's gap-filling CV                       python -m sparse_panel.interp_cv
  evaluate.py          the paper's final fit and evaluation table       python -m sparse_panel.evaluate
  paths.py             data/, results/, outputs/
paper/                 Quarto write-up -> PDF for alphaXiv
  index.qmd            the paper; code cells import sparse_panel
  _quarto.yml          PDF format; renders to paper/_output/
  references.bib, _wide-tables.lua, thumbnail.png
  notes/               research notes from the analysis (summary, decisions, todolist)
results/               tracked CV and evaluation results; the paper reads cv_results.csv
huggingface/
  space/               Gradio app (app.py), Space card and requirements
  dataset/             precompute.py (builds the demo's tables in outputs/) and the dataset card
  model/               model card and a runnable example (model.py is copied from sparse_panel/)
  publish.sh           upload any of dataset / space / model
tests/                 checks that the paper's sparse data is reproduced exactly
data/, outputs/        downloaded data and generated tables (not in git)
```

## Common tasks

```bash
uv sync                                                    # library + app + paper + dev groups

QUARTO_PYTHON=.venv/bin/python quarto render paper         # paper PDF (~1 min; downloads data on first run)
uv run python -m sparse_panel.cv                           # re-run the paper's CV (slow; rewrites results/)

uv run python huggingface/dataset/precompute.py           # demo tables in outputs/ (~30 s)
uv run python huggingface/space/app.py                     # app on http://127.0.0.1:7860
huggingface/publish.sh "message" space                     # publish (targets: dataset, space, model; default all)

uv run pytest -q
```

The paper's sparse data depends on the order the station files are stacked in, which is pinned in
`sparse_panel.data.PAPER_STATION_ORDER`; the test fails if the data stops matching.

Project notes for the demo and repo: `summary.md`, `decisions.md`, `todolist.txt`.

Data: Beijing Multi-Site Air Quality, Song Chen (2017), UCI Machine Learning Repository, CC BY 4.0.
