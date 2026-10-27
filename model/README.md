---
license: cc-by-4.0
library_name: pytorch
tags:
  - time-series
  - panel-data
  - missing-data
  - smoothing
  - imputation
  - air-quality
  - pytorch
datasets:
  - prteek/sparse-panel-demo
---

# Prevailing level network for sparse, gappy panels

A small PyTorch network that estimates a smooth **prevailing level** for every location and day of a panel,
for two related quantities at once, when the observations are noisy daily means with long gaps and some series
missing entirely. It was developed for the blog post
[Smooth Levels from Sparse, Gappy Panels](https://huggingface.co/blog/prteek/sparse-panel-modelling) and powers the
[sparse panel demo Space](https://huggingface.co/spaces/prteek/sparse-panel-demo).

**This repo holds the code and the training recipe, not trained weights.** The network learns an embedding for
each location and a time basis over a fixed date range, so a fitted model only applies to the panel it was fitted
on. To use it, fit it to your own panel: on the 12-station, four-year Beijing data a fit takes a few seconds on a
CPU.

## Files

- `model.py`: the encoding, the network, the loss and `fit` / `predict`
- `example.py`: fits the network to a small synthetic panel and predicts every cell (runs as is)

## Model

Each row is one location and day, with up to two targets \\( y_1, y_2 \\) (log values, standardised).

$$
\begin{aligned}
x &= \big[\, e_{\text{area}} + e_{\text{station}},\ b(t) \,\big] \\
h &= \operatorname{SiLU}\!\big(W_2 \operatorname{SiLU}(W_1 x)\big) \\
\mu_1 &= w_1^\top h, \quad \mu_2 = w_2^\top h
\end{aligned}
$$

- **Location**: an 8-dim area embedding plus a station offset that starts near zero and is penalised, so a
  station with little data falls back to its area.
- **Time**: fixed Gaussian bumps \\( b_k(t) = \exp\big(-\tfrac12((t - c_k)/w)^2\big) \\) with centres every \\( w \\) days, so the
  level is a smooth function of time.
- **Shared trunk, two heads**: two layers of 64 with SiLU, then one linear head per target. A day where only one
  target was observed still trains everything the other depends on.

**Loss.** An observed value is a mean of \\( n \\) readings, so its variance around the level is \\( \sigma_j^2 / n + \tau_j^2 \\),
with \\( \sigma_j \\), \\( \tau_j \\) learned per target. The Gaussian negative log-likelihood is averaged over observed cells
only: a missing target is **masked**, not dropped. Added to it:

- a **curvature penalty** \\( \lambda \cdot \overline{(\mu_{t+1} - 2\mu_t + \mu_{t-1})^2} \\) over every location's
  predicted series on the **whole** day grid, observed or not, which keeps the level smooth through gaps;
- a small L2 penalty on the station offsets.

## Training recipe

Full-batch Adam with a cosine learning-rate schedule. The settings below were chosen by look-ahead
cross-validation in the blog post:

| Setting | Value |
|---|---|
| curvature penalty \\( \lambda \\) (`rough`) | 1e5 |
| bump spacing \\( w \\) (`spacing`) | 30 days |
| hidden width | 64 |
| learning rate | 5e-3 |
| weight decay | 1e-4 |
| station offset L2 (`station_l2`) | 1e-3 |
| schedule length | 1000 |
| training steps | 100 |
| seed | 0 |

For a single-target model, pass `use=["PM2.5"]` (or the other target) to `fit`.

## Input format

Long format, one row per location × date × target that has data:

| column | meaning |
|---|---|
| `station` | location id |
| `area` | group the location belongs to (it falls back to this when its own data is thin) |
| `date` | daily timestamp |
| `pollutant` | target name, one of `model.TARGETS` |
| `value` | the daily mean (> 0; modelled on the log scale) |
| `n` | number of readings behind that mean (weights the observation) |

The two target names are set in `model.TARGETS` (`["PM2.5", "PM10"]`); change them for other quantities.

```python
from model import Encoding, fit, predict

enc = Encoding.fit(panel, end=panel["date"].max())
model = fit(enc, enc.wide(panel), CFG, steps=100)
level = predict(model, enc, enc.grid())   # every location x day, observed or not
```

## Limitations

- **Not transferable.** Location embeddings and the time basis are specific to one panel and date range. Refit
  for new locations or dates.
- **Smoothness is a choice.** Cross-validation that scores against noisy readings rewards following short
  episodes, so it tends to under-smooth; the blog post discusses this. Treat `rough` and `spacing` as decisions about
  how smooth the level should be.
- **Log-scale bias.** The model is fitted to logs of means over few readings, so its level sits slightly below a
  moving average of full data (Jensen's inequality).
- **A GAM is a strong alternative.** In the blog post, a mixed-effects GAM with hand-set smoothness does as well or
  better.
- **Memory.** Training is full batch and the penalty covers every location × day, so memory grows with panel size.

## Data used in the blog post

Beijing Multi-Site Air Quality, Song Chen (2017), UCI Machine Learning Repository,
[doi:10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0. The demo's sparse readings and predictions
are in the dataset [prteek/sparse-panel-demo](https://huggingface.co/datasets/prteek/sparse-panel-demo).
