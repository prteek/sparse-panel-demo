"""Fit every scenario offline and write the tables the app reads.

One scenario per global thinning level (LAMBDAS). In each, stations are thinned according to their label
(sparse_panel.data), and three estimators are fitted: the joint network, one single-target network per pollutant,
and the ±15-day mean. Everything is scored against the 31-day MA of the full data.

Writes to outputs/:
  labels.parquet       station, area, label, dropped
  truth.parquet        station, pollutant, date, value          (31-day MA of the full data)
  readings.parquet     lam, station, pollutant, date, value, n  (the sparse data each scenario sees)
  gaps.parquet         station, pollutant, date                 (gap blocks; the same in every scenario)
  predictions.parquet  lam, estimator, station, pollutant, date, pred
  metrics.parquet      lam, label, pollutant, estimator, day_type, days, error, bias
                       (day_type: all / observed / unobserved / gap; for 'dropped series' pollutant is 'dropped')
  meta.json            settings used
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sparse_panel.baselines import KEY, moving_average, smooth_truth
from sparse_panel.data import AREA, SEED, assign_labels, daily, download, load_hourly, thin
from sparse_panel.model import TARGETS, Encoding, fit, predict

HERE = Path(__file__).parent
LAMBDAS = [0.5, 1.2, 3.0]
# Settings chosen by look-ahead CV in the post (posts/beijing_air_quality, cv.py / evaluate.py)
CFG = dict(rough=1e5, spacing=30, lr=5e-3, hidden=64, weight_decay=1e-4, station_l2=1e-3, schedule=1000, seed=0)
STEPS = 100


def day_types(cells: pd.DataFrame, readings: pd.DataFrame, gaps: pd.DataFrame) -> pd.Series:
    c = cells.merge(readings[KEY].assign(obs=True), on=KEY, how="left").merge(
        gaps.assign(gap=True), on=KEY, how="left")
    return pd.Series(np.select([c["obs"].notna(), c["gap"].notna()], ["observed", "gap"], "unobserved"),
                     index=cells.index)


def score(pred: pd.DataFrame, truth: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    r = pred.merge(truth, on=KEY).merge(labels[["station", "label", "dropped"]], on="station")
    # a 'dropped series' station is scored on its dropped pollutant only (the other one keeps its data)
    dropped = r["label"] == "dropped series"
    r = r[~dropped | (r["pollutant"] == r["dropped"])]
    r.loc[r["label"] == "dropped series", "pollutant"] = "dropped"
    r["e"] = np.log(r["pred"]) - np.log(r["value"])
    r = r.dropna(subset=["e"])
    rows = []
    for day_type, d in [("all", r), *r.groupby("day_type")]:
        g = d.groupby(["lam", "label", "pollutant", "estimator"])["e"]
        rows.append(pd.DataFrame({"days": g.size(), "error": g.apply(lambda e: e.abs().mean()), "bias": g.mean()})
                    .reset_index().assign(day_type=day_type))
    return pd.concat(rows, ignore_index=True)


def main():
    torch.set_num_threads(6)
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    hourly = load_hourly(download(HERE / "data"))
    full = daily(hourly)
    truth = smooth_truth(full)
    labels = assign_labels(SEED)
    print(labels.to_string(index=False))

    readings, preds, gaps = [], [], None
    for lam in LAMBDAS:
        sparse, gaps = thin(hourly, lam, labels, SEED)
        enc = Encoding.fit(sparse, end=full["date"].max())
        w, grid = enc.wide(sparse), enc.grid()
        cells = grid.merge(pd.DataFrame({"pollutant": TARGETS}), how="cross")
        est = {
            "joint": predict(fit(enc, w, CFG, STEPS), enc, grid),
            "single-target": pd.concat([predict(fit(enc, w, CFG, STEPS, use=[t]), enc, grid).query("pollutant == @t")
                                        for t in TARGETS]),
            "±15-day mean": moving_average(sparse, cells, AREA),
        }
        types = cells.assign(day_type=day_types(cells, sparse, gaps))
        for name, p in est.items():
            preds.append(p[KEY + ["pred"]].merge(types, on=KEY).assign(lam=lam, estimator=name))
        readings.append(sparse.drop(columns="area").assign(lam=lam))
        print(f"lam={lam}: {len(sparse):,} readings")

    preds = pd.concat(preds, ignore_index=True)
    metrics = score(preds, truth, labels)
    preds = preds.drop(columns="day_type")

    labels.to_parquet(out / "labels.parquet", index=False)
    truth.to_parquet(out / "truth.parquet", index=False)
    pd.concat(readings, ignore_index=True).to_parquet(out / "readings.parquet", index=False)
    gaps.to_parquet(out / "gaps.parquet", index=False)
    preds.to_parquet(out / "predictions.parquet", index=False)
    metrics.to_parquet(out / "metrics.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(
        dict(lambdas=LAMBDAS, cfg=CFG, steps=STEPS, seed=SEED, created=pd.Timestamp.now().isoformat(timespec="seconds")),
        indent=2))

    summary = metrics[metrics["day_type"] == "all"].pivot_table(
        index=["lam", "label"], columns="estimator", values="error").round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
