"""Fit the prevailing-level network to a panel of your own and predict a smooth level for every cell.

Input is long format, one row per station x date x pollutant that has data:
  station, area, date, pollutant (one of model.TARGETS), value (> 0, a daily mean), n (readings behind it)
Here a small synthetic panel stands in for real data.
"""
import numpy as np
import pandas as pd

from model import TARGETS, Encoding, fit, predict

# Settings chosen by look-ahead cross-validation in the blog post
CFG = dict(rough=1e5, spacing=30, lr=5e-3, hidden=64, weight_decay=1e-4, station_l2=1e-3, schedule=1000, seed=0)
STEPS = 100


def synthetic_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2020-01-01", periods=365)
    stations = {"s1": "urban", "s2": "urban", "s3": "suburban", "s4": "suburban"}
    t = np.arange(len(days))
    rows = []
    for s, area in stations.items():
        level = 4 + 0.5 * np.sin(2 * np.pi * t / 365) + (0.3 if area == "urban" else 0) + rng.normal(0, 0.1)
        for j, pollutant in enumerate(TARGETS):
            n = rng.poisson(1.2, len(days))  # hours observed per day; 0 means no reading
            keep = n > 0
            value = np.exp(level + 0.4 * j + rng.normal(0, 0.5 / np.sqrt(np.maximum(n, 1))))
            rows.append(pd.DataFrame({"station": s, "area": area, "date": days[keep], "pollutant": pollutant,
                                      "value": value[keep], "n": n[keep]}))
    return pd.concat(rows, ignore_index=True)


panel = synthetic_panel()
enc = Encoding.fit(panel, end=panel["date"].max())
model = fit(enc, enc.wide(panel), CFG, STEPS)            # joint model; use=["PM2.5"] for a single-target one
level = predict(model, enc, enc.grid())                  # every station x day, observed or not
print(level.head())
print(f"{len(panel)} readings in, {len(level)} predicted cells out")
