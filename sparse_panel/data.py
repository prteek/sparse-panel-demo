"""Beijing multi-site air quality data: download, daily means, and the two ways of thinning it to sparse readings.

- Blog post (`prepare_blog_data`, `thin_uniform`): every station and pollutant keeps a Poisson(BLOG_LAMBDA) random subset
  of its valid hours per day, plus BLOG_GAP_BLOCKS runs of 20-60 empty days. Writes data/daily_full.parquet,
  data/daily_sparse.parquet and data/gaps.parquet, which the blog post and its CV scripts read.
- Demo (`assign_labels`, `thin_by_label`): every station gets one label, which decides how its readings are thinned
  relative to the global setting `lam` (Poisson mean of hours kept per station-day and pollutant):

  - regular:        Poisson(lam) hours, plus GAP_BLOCKS gap blocks of 20-60 days per pollutant
  - sparse:         Poisson(SPARSE_FACTOR * lam) hours, same gaps as regular
  - very sparse:    as regular, then only VERY_SPARSE_KEEP of the observed days kept
  - long gap:       as regular, but the gap blocks are 90-120 days
  - dropped series: one pollutant removed entirely; the other is treated as regular

  All random draws are fixed per cell (seeded independently of `lam`), so scenarios differ only by `lam`:
  a higher `lam` keeps a superset of the hours a lower one keeps, and gaps sit in the same place.
"""
import glob
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_panel.paths import DATA

URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
CSV_DIR = "PRSA_Data_20130301-20170228"
POLLUTANTS = ["PM2.5", "PM10"]
SEED = 0

# Urban core vs suburban monitoring sites (the "area" level the network borrows from)
AREA = {"Dongsi": "urban", "Tiantan": "urban", "Guanyuan": "urban", "Wanshouxigong": "urban", "Nongzhanguan": "urban",
        "Gucheng": "urban", "Aotizhongxin": "urban", "Wanliu": "urban",
        "Changping": "suburban", "Dingling": "suburban", "Huairou": "suburban", "Shunyi": "suburban"}


# ---- shared: download and daily means -------------------------------------------------------------------------------

def download(data_dir: Path = DATA) -> Path:
    """Fetch and unpack the UCI archive once; returns the folder of per-station hourly CSVs."""
    out = data_dir / CSV_DIR
    if out.exists():
        return out
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "beijing.zip"
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extract("PRSA2017_Data_20130301-20170228.zip", data_dir)
    with zipfile.ZipFile(data_dir / "PRSA2017_Data_20130301-20170228.zip") as z:
        z.extractall(data_dir)
    return out


def load_hourly(csv_dir: Path, station_order=None) -> pd.DataFrame:
    """Hourly readings of all stations, stacked in `station_order` (default: alphabetical)."""
    files = sorted(glob.glob(str(csv_dir / "*.csv")))
    if station_order is not None:
        by_station = {Path(f).name.split("_")[2]: f for f in files}
        files = [by_station[s] for s in station_order]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    return df[["station", "date", "hour"] + POLLUTANTS]


def daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Long format: station, date, pollutant, value (mean over valid hours), n (hours), area."""
    long = hourly.melt(id_vars=["station", "date", "hour"], value_vars=POLLUTANTS, var_name="pollutant").dropna()
    out = long.groupby(["station", "date", "pollutant"])["value"].agg(value="mean", n="count").reset_index()
    return out.assign(area=out["station"].map(AREA))


# ---- blog post: uniform thinning -------------------------------------------------------------------------------------

BLOG_LAMBDA = 1.2
BLOG_GAP_BLOCKS, BLOG_GAP_MIN, BLOG_GAP_MAX = 2, 20, 60
# The order the station files were stacked in when the blog post's sparse data was drawn (the original code took
# glob's filesystem order). The shuffle below depends on it, so it is pinned to reproduce the blog post exactly.
BLOG_STATION_ORDER = ["Gucheng", "Huairou", "Tiantan", "Changping", "Guanyuan", "Nongzhanguan", "Wanliu", "Dongsi",
                       "Wanshouxigong", "Aotizhongxin", "Dingling", "Shunyi"]


def thin_uniform(hourly: pd.DataFrame, rng):
    """Keep a Poisson(BLOG_LAMBDA) random subset of valid hours per station x day x pollutant, then cut gap blocks."""
    long = hourly.melt(id_vars=["station", "date", "hour"], value_vars=POLLUTANTS, var_name="pollutant").dropna()
    long = long.sample(frac=1, random_state=SEED)  # random order within each cell
    keys = long[["station", "date", "pollutant"]].drop_duplicates()
    keys["k"] = rng.poisson(BLOG_LAMBDA, len(keys))
    long = long.merge(keys, on=["station", "date", "pollutant"])
    long["rank"] = long.groupby(["station", "date", "pollutant"]).cumcount()
    kept = long[long["rank"] < long["k"]]

    days = pd.date_range(hourly["date"].min(), hourly["date"].max())
    gaps = []
    for st in sorted(hourly["station"].unique()):
        for pol in POLLUTANTS:
            for _ in range(BLOG_GAP_BLOCKS):
                length = rng.integers(BLOG_GAP_MIN, BLOG_GAP_MAX + 1)
                start = rng.integers(0, len(days) - length)
                gaps.append(pd.DataFrame({"station": st, "pollutant": pol, "date": days[start:start + length]}))
    gaps = pd.concat(gaps).drop_duplicates().assign(in_gap=True)
    kept = kept.merge(gaps, on=["station", "date", "pollutant"], how="left")
    kept = kept[kept["in_gap"].isna()]
    out = kept.groupby(["station", "date", "pollutant"])["value"].agg(value="mean", n="count").reset_index()
    return out.assign(area=out["station"].map(AREA)), gaps.drop(columns="in_gap")


def prepare_blog_data(data_dir: Path = DATA, force: bool = False) -> None:
    """Write daily_full, daily_sparse and gaps parquet files for the blog post (skipped if they exist)."""
    if not force and (data_dir / "daily_sparse.parquet").exists():
        return
    hourly = load_hourly(download(data_dir), station_order=BLOG_STATION_ORDER)
    full = daily(hourly)
    sparse, gaps = thin_uniform(hourly, np.random.default_rng(SEED))
    full.to_parquet(data_dir / "daily_full.parquet", index=False)
    sparse.to_parquet(data_dir / "daily_sparse.parquet", index=False)
    gaps.to_parquet(data_dir / "gaps.parquet", index=False)


# ---- demo: label-dependent thinning ---------------------------------------------------------------------------------

LABELS = ["regular", "sparse", "very sparse", "long gap", "dropped series"]
DEGRADED = LABELS[1:]  # one urban and one suburban station each; the remaining urban stations are regular
SPARSE_FACTOR = 0.4
VERY_SPARSE_KEEP = 0.15
GAP_BLOCKS = 2
GAP_DAYS = {"long gap": (90, 120)}
DEFAULT_GAP_DAYS = (20, 60)


def assign_labels(seed: int = SEED) -> pd.DataFrame:
    """One label per station: each degraded label gets one urban and one suburban station.

    `dropped` names the removed pollutant for 'dropped series' stations (PM2.5 at one, PM10 at the other).
    """
    rng = np.random.default_rng([seed, 1])
    urban = rng.permutation(sorted(s for s, a in AREA.items() if a == "urban"))
    suburban = rng.permutation(sorted(s for s, a in AREA.items() if a == "suburban"))
    assert len(suburban) == len(DEGRADED), "expects exactly one suburban station per degraded label"
    rows = [(s, l) for s, l in zip(suburban, rng.permutation(DEGRADED))]
    urban_labels = list(DEGRADED) + ["regular"] * (len(urban) - len(DEGRADED))
    rows += [(s, l) for s, l in zip(urban, rng.permutation(urban_labels))]
    labels = pd.DataFrame(rows, columns=["station", "label"]).assign(area=lambda d: d["station"].map(AREA))
    dropped = labels.loc[labels["label"] == "dropped series", "station"].sort_values().to_list()
    labels["dropped"] = labels["station"].map(dict(zip(dropped, rng.permutation(POLLUTANTS))))
    return labels.sort_values(["label", "station"], key=lambda c: c.map(LABELS.index) if c.name == "label" else c,
                              ignore_index=True)


def poisson_ppf(u: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Inverse Poisson CDF, so one uniform draw per cell gives a hour count that grows with lam."""
    out = np.empty(len(u), dtype=int)
    for l in np.unique(lam):
        pmf = np.empty(64)
        pmf[0] = np.exp(-l)
        for k in range(1, len(pmf)):
            pmf[k] = pmf[k - 1] * l / k
        m = lam == l
        out[m] = np.searchsorted(np.cumsum(pmf), u[m], side="right")
    return out


def gap_blocks(stations, days: pd.DatetimeIndex, labels: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng([seed, 2])
    label_of = labels.set_index("station")["label"]
    out = []
    for st in sorted(stations):
        lo, hi = GAP_DAYS.get(label_of[st], DEFAULT_GAP_DAYS)
        for pol in POLLUTANTS:
            for _ in range(GAP_BLOCKS):
                length = rng.integers(lo, hi + 1)
                start = rng.integers(0, len(days) - length)
                out.append(pd.DataFrame({"station": st, "pollutant": pol, "date": days[start:start + length]}))
    return pd.concat(out).drop_duplicates(ignore_index=True)


def thin_by_label(hourly: pd.DataFrame, lam: float, labels: pd.DataFrame, seed: int = SEED):
    """Sparse daily readings for one global thinning level. Returns (sparse, gaps)."""
    long = hourly.melt(id_vars=["station", "date", "hour"], value_vars=POLLUTANTS, var_name="pollutant").dropna()
    # fixed random order of hours within each cell
    long = long.assign(order=np.random.default_rng([seed, 3]).random(len(long))).sort_values(
        ["station", "date", "pollutant", "order"])
    long["rank"] = long.groupby(["station", "date", "pollutant"]).cumcount()

    cells = long[["station", "date", "pollutant"]].drop_duplicates(ignore_index=True)
    rng = np.random.default_rng([seed, 4])
    u_hours, u_keep = rng.random(len(cells)), rng.random(len(cells))
    label = cells["station"].map(labels.set_index("station")["label"])
    cells["k"] = poisson_ppf(u_hours, np.where(label == "sparse", SPARSE_FACTOR * lam, lam))
    cells = cells[~((label == "very sparse") & (u_keep >= VERY_SPARSE_KEEP))]
    dropped = labels.dropna(subset=["dropped"])
    cells = cells.merge(dropped[["station", "dropped"]], on="station", how="left")
    cells = cells[cells["pollutant"] != cells["dropped"]].drop(columns="dropped")

    days = pd.date_range(hourly["date"].min(), hourly["date"].max())
    gaps = gap_blocks(hourly["station"].unique(), days, labels, seed)
    cells = cells.merge(gaps.assign(in_gap=True), on=["station", "date", "pollutant"], how="left")
    cells = cells[cells["in_gap"].isna()].drop(columns="in_gap")

    kept = long.merge(cells, on=["station", "date", "pollutant"])
    kept = kept[kept["rank"] < kept["k"]]
    out = kept.groupby(["station", "date", "pollutant"])["value"].agg(value="mean", n="count").reset_index()
    return out.assign(area=out["station"].map(AREA)), gaps


if __name__ == "__main__":
    prepare_blog_data(force=True)
    sparse = pd.read_parquet(DATA / "daily_sparse.parquet")
    print(f"wrote blog post data to {DATA}: {len(sparse):,} sparse station-day-pollutant readings")
