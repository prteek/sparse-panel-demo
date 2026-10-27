# sparse-panel-modelling

Smooth levels from sparse, gappy panels, on hourly PM2.5 and PM10 from 12 Beijing stations. This repo holds
everything for the blog post [Smooth Levels from Sparse, Gappy Panels](https://huggingface.co/blog/prteek/sparse-panel-modelling)
on Hugging Face: the write-up, the code it runs, and the Hugging Face artefacts built from the same code.

**Blog:** https://huggingface.co/blog/prteek/sparse-panel-modelling · **Live app:** https://huggingface.co/spaces/prteek/sparse-panel-demo · **Model:** https://huggingface.co/prteek/sparse-panel-demo · **Data:** https://huggingface.co/datasets/prteek/sparse-panel-demo

## Layout

```
sparse_panel/          shared library, used by everything below
  data.py              UCI download, daily means; the blog's uniform thinning and the demo's label-based thinning
  model.py             the joint two-target network (fit / predict)
  baselines.py         31-day truth, ±15-day mean with area fallback
  cv.py                the blog's look-ahead CV and best_config()       python -m sparse_panel.cv
  interp_cv.py         the blog's gap-filling CV                        python -m sparse_panel.interp_cv
  evaluate.py          the blog's final fit and evaluation table        python -m sparse_panel.evaluate
  paths.py             data/, results/, outputs/
results/               tracked CV and evaluation results; the blog post reads cv_results.csv
blog/                  Quarto write-up -> markdown for the Hugging Face article editor
  index.qmd            the blog post; renders to sparse-panel-modelling.md, figures to assets/
  _hf-markdown.lua     writes inline maths as \\( ... \\) for the article editor
  thumbnail.png        the article's cover image (hand-picked, not generated)
space/                 Gradio app (app.py), Space card and requirements
dataset/               precompute.py (builds the demo's tables in outputs/) and the dataset card
model/                 model card and a runnable example (model.py is copied from sparse_panel/)
publish.sh             upload any of dataset / space / model to Hugging Face
wait_for_space.py      CI: wait until the Space runs the uploaded app
tests/                 checks that the blog's sparse data is reproduced exactly
data/, outputs/        downloaded data and generated tables (not in git)
```

## Common tasks

```bash
uv sync                                                    # library + app + blog + dev groups

QUARTO_PYTHON=.venv/bin/python quarto render blog/index.qmd # blog markdown + assets/ figures (~1 min; downloads data on first run)
uv run python -m sparse_panel.cv                           # re-run the blog's CV (slow; rewrites results/)

uv run python dataset/precompute.py                        # demo tables in outputs/ (~30 s)
uv run python space/app.py                                 # app on http://127.0.0.1:7860
./publish.sh "message" space                               # publish (targets: dataset, space, model; default all)

uv run pytest -q
```

## Publishing from CI

Pushes to `main` publish to Hugging Face through `.github/workflows/` (each can also be run by hand), using the
`HF_TOKEN` repository secret:

- `publish-space.yml`: when `space/` changes; uploads the app and waits until the Space runs it
  (restarting it once if the upload didn't trigger a rebuild)
- `publish-model.yml`: when `sparse_panel/model.py` or `model/` changes; runs the example, then uploads
- `publish-dataset.yml`: when the dataset card changes; uploads the card only. The tables are rebuilt with
  `precompute.py` and published by hand (`./publish.sh "message" dataset`), so the published predictions
  stay exactly the ones fitted and checked locally.

The blog's sparse data depends on the order the station files are stacked in, which is pinned in
`sparse_panel.data.BLOG_STATION_ORDER`; the test fails if the data stops matching.

Project notes for the blog post, the demo and the repo: `summary.md`, `decisions.md`, `todolist.txt`.

Data: Beijing Multi-Site Air Quality, Song Chen (2017), UCI Machine Learning Repository, CC BY 4.0.
