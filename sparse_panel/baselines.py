"""The benchmark (31-day MA of the full data) and the simple ±15-day mean estimator, as in the post."""
import numpy as np
import pandas as pd

KEY = ["station", "date", "pollutant"]


def smooth_truth(full: pd.DataFrame, window: int = 31) -> pd.DataFrame:
    """The 'true prevailing level': centred rolling mean of the full-data daily means on the log scale."""
    out = []
    for (st, pol), g in full.sort_values("date").groupby(["station", "pollutant"]):
        s = np.log(g.set_index("date")["value"]).asfreq("D").rolling(window, center=True, min_periods=window // 2).mean()
        out.append(pd.DataFrame({"station": st, "pollutant": pol, "date": s.index, "value": np.exp(s.values)}))
    return pd.concat(out, ignore_index=True).dropna()


def window_mean(sparse: pd.DataFrame, keys: pd.DataFrame, half_window: int = 15) -> pd.DataFrame:
    """Hour-weighted mean of log readings within ±half_window days of each day, per station x pollutant."""
    out = []
    for (st, pol), g in sparse.groupby(["station", "pollutant"]):
        s = g.set_index("date")
        lw = (np.log(s["value"]) * s["n"]).asfreq("D").fillna(0)
        n = s["n"].asfreq("D").fillna(0)
        w = 2 * half_window + 1
        lvl = np.exp(lw.rolling(w, center=True, min_periods=1).sum() / n.rolling(w, center=True, min_periods=1).sum())
        out.append(pd.DataFrame({"station": st, "pollutant": pol, "date": lvl.index, "pred": lvl.values}))
    return keys.merge(pd.concat(out), on=KEY, how="left")


def moving_average(sparse: pd.DataFrame, keys: pd.DataFrame, area_of: dict) -> pd.DataFrame:
    """±15-day mean of the station's own readings; where it has none in the window, the same over its area."""
    station = window_mean(sparse, keys)
    area_daily = (sparse.assign(lw=np.log(sparse["value"]) * sparse["n"])
                  .groupby(["area", "date", "pollutant"]).agg(lw=("lw", "sum"), n=("n", "sum")).reset_index()
                  .assign(value=lambda d: np.exp(d["lw"] / d["n"])).rename(columns={"area": "station"}))
    area = window_mean(area_daily, keys.assign(station=keys["station"].map(area_of)))
    station["pred"] = station["pred"].fillna(pd.Series(area["pred"].to_numpy(), index=station.index))
    return station
