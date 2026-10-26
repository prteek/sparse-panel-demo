"""Gap-filling CV: hide random blocks inside the observed period and score how well they are filled.

For each of N_SPLITS splits, every station x pollutant loses BLOCKS random BLOCK_DAYS-day blocks of the sparse data
(outside the original gap blocks, which stay untouched as the final test). The model is fitted on the rest of the
whole period; the hidden blocks are scored against the sparse readings (selection) and the true smooth level.

Run: python -m sparse_panel.interp_cv [--workers N]. Writes results/interp_cv_results.csv and
results/interp_cv_summary.md.
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from sparse_panel.baselines import smooth_truth
from sparse_panel.cv import CHECKPOINTS, FIXED, GRID, load, score
from sparse_panel.model import KEY, TARGETS, Encoding, fit, predict
from sparse_panel.paths import DATA, RESULTS

N_SPLITS, BLOCKS, BLOCK_DAYS = 3, 2, 30


def hidden_blocks(sparse: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(100 + seed)
    gaps = pd.read_parquet(DATA / "gaps.parquet").assign(gap=True)
    days = pd.date_range(sparse["date"].min(), sparse["date"].max())
    rows = []
    for st in sorted(sparse["station"].unique()):
        for pol in TARGETS:
            g = set(gaps.loc[(gaps["station"] == st) & (gaps["pollutant"] == pol), "date"])
            placed = 0
            while placed < BLOCKS:
                start = rng.integers(0, len(days) - BLOCK_DAYS)
                block = days[start:start + BLOCK_DAYS]
                if not g.intersection(block):
                    rows.append(pd.DataFrame({"station": st, "pollutant": pol, "date": block}))
                    g.update(block)
                    placed += 1
    return pd.concat(rows).drop_duplicates().assign(hidden=True)


def run_config(i_cfg):
    torch.set_num_threads(1)
    i, cfg = i_cfg
    cfg = {**FIXED, **cfg}
    sparse, full = load()
    level = smooth_truth(full)
    rows = []
    for split in range(N_SPLITS):
        hid = hidden_blocks(sparse, split)
        tagged = sparse.merge(hid, on=["station", "pollutant", "date"], how="left")
        train, val_sparse = tagged[tagged["hidden"].isna()], tagged[tagged["hidden"].notna()]
        val_level = level.merge(hid, on=["station", "pollutant", "date"])
        enc = Encoding.fit(train, end=sparse["date"].max())
        keys = hid[KEY].drop_duplicates()

        def evaluate(step, model):
            pred = predict(model, enc, keys)
            rows.append(dict(config=i, split=split, step=step, rough=cfg["rough"], spacing=cfg["spacing"],
                             **score(val_sparse, pred, "sparse"), **score(val_level, pred, "level")))

        fit(enc, enc.wide(train), cfg, max(CHECKPOINTS), on_checkpoint=evaluate, checkpoints=CHECKPOINTS)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    load()  # prepare the data once before the workers start
    with Pool(args.workers) as pool:
        res = pd.DataFrame([r for rows in pool.map(run_config, list(enumerate(GRID))) for r in rows])
    res.to_csv(RESULTS / "interp_cv_results.csv", index=False)
    summ = (res.groupby(["config", "rough", "spacing", "step"])
            .agg(sparse_mean=("sparse_mean", "mean"), sparse_sd=("sparse_mean", "std"),
                 **{f"sparse_{t}": (f"sparse_{t}", "mean") for t in TARGETS},
                 level_mean=("level_mean", "mean"))
            .reset_index().sort_values("sparse_mean"))
    lines = [f"# Gap-filling CV ({pd.Timestamp.now():%Y-%m-%d %H:%M})", "",
             f"{N_SPLITS} splits, each hiding {BLOCKS} random {BLOCK_DAYS}-day blocks per station × pollutant; "
             f"{len(GRID)} configs × {len(CHECKPOINTS)} step counts. Hour-weighted mean absolute log error on hidden "
             "sparse readings (`sparse_*`, used for selection) and against the true 31-day level (`level_mean`).", "",
             "## Top 10", "", summ.head(10).drop(columns="config").round(4).to_markdown(index=False, floatfmt="g"), "",
             "## Worst 3", "", summ.tail(3).drop(columns="config").round(4).to_markdown(index=False, floatfmt="g")]
    (RESULTS / "interp_cv_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
