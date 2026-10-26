"""Joint two-target network for a smooth 'prevailing' pollution level per station and day.

Copied from the regression_room post (posts/beijing_air_quality/model.py):
- one row per station x day with up to two targets (log PM2.5, log PM10); a missing target is masked, not dropped;
- location = area embedding + small station offset (station falls back to its area when data is thin);
- time enters only through fixed Gaussian bumps, so the level is a smooth function of time;
- loss = Gaussian NLL of an observed daily mean of n hours, variance sigma^2 / n + tau^2;
- a roughness penalty on the second difference of every station's predicted series over the full day grid.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

TARGETS = ["PM2.5", "PM10"]
KEY = ["station", "date"]


@dataclass
class Encoding:
    stations: list
    areas: list
    station_area: dict
    start: pd.Timestamp
    n_days: int
    y_mean: np.ndarray
    y_std: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, end: pd.Timestamp) -> "Encoding":
        start = df["date"].min()
        logv = np.log(df["value"])
        return cls(stations=sorted(df["station"].unique()), areas=sorted(df["area"].unique()),
                   station_area=df.drop_duplicates("station").set_index("station")["area"].to_dict(),
                   start=start, n_days=(end - start).days + 1,
                   y_mean=np.array([logv[df["pollutant"] == t].mean() for t in TARGETS]),
                   y_std=np.array([logv[df["pollutant"] == t].std() for t in TARGETS]))

    def wide(self, df: pd.DataFrame) -> pd.DataFrame:
        w = df.pivot_table(index=KEY, columns="pollutant", values=["value", "n"], aggfunc="first")
        w.columns = [f"{v}_{p}" for v, p in w.columns]
        for t in TARGETS:
            for v in ["value", "n"]:
                if f"{v}_{t}" not in w:
                    w[f"{v}_{t}"] = np.nan
        return w.reset_index()

    def inputs(self, keys: pd.DataFrame) -> dict:
        s_idx = {s: i for i, s in enumerate(self.stations)}
        a_idx = {a: i for i, a in enumerate(self.areas)}
        return {"station": torch.tensor(keys["station"].map(s_idx).to_numpy()),
                "area": torch.tensor(keys["station"].map(self.station_area).map(a_idx).to_numpy()),
                "day": torch.tensor((keys["date"] - self.start).dt.days.to_numpy())}

    def targets(self, w: pd.DataFrame, use=TARGETS):
        val = w[[f"value_{t}" for t in TARGETS]].to_numpy(float)
        n = w[[f"n_{t}" for t in TARGETS]].to_numpy(float)
        mask = ~np.isnan(val)
        mask[:, [t not in use for t in TARGETS]] = False  # single-target models ignore the other pollutant
        y = (np.log(np.where(mask, val, 1.0)) - self.y_mean) / self.y_std
        return (torch.tensor(np.where(mask, y, 0.0), dtype=torch.float32), torch.tensor(mask),
                torch.tensor(np.where(mask, n, 1.0), dtype=torch.float32))

    def to_value(self, mu: torch.Tensor) -> np.ndarray:
        return np.exp(mu.detach().numpy() * self.y_std + self.y_mean)

    def grid(self) -> pd.DataFrame:
        days = pd.date_range(self.start, periods=self.n_days)
        return pd.DataFrame({"station": self.stations}).merge(pd.DataFrame({"date": days}), how="cross")


class TimeBasis(nn.Module):
    def __init__(self, n_days: int, spacing: int):
        super().__init__()
        self.register_buffer("centres", torch.arange(-spacing, n_days + 2 * spacing, spacing, dtype=torch.float32))
        self.width = float(spacing)

    def forward(self, day):
        return torch.exp(-0.5 * ((day.float()[:, None] - self.centres[None, :]) / self.width) ** 2)


class PrevailingNet(nn.Module):
    def __init__(self, enc: Encoding, dim_loc: int = 8, hidden: int = 64, spacing: int = 30):
        super().__init__()
        self.time = TimeBasis(enc.n_days, spacing)
        self.area = nn.Embedding(len(enc.areas), dim_loc)
        nn.init.normal_(self.area.weight, std=0.1)  # small init so the area vectors are actually learned
        self.station = nn.Embedding(len(enc.stations), dim_loc)
        nn.init.normal_(self.station.weight, std=0.01)
        self.trunk = nn.Sequential(nn.Linear(dim_loc + len(self.time.centres), hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in TARGETS])
        self.log_sigma = nn.Parameter(torch.zeros(len(TARGETS)))
        self.log_tau = nn.Parameter(torch.full((len(TARGETS),), -1.0))
        self.register_buffer("area_of", torch.tensor([enc.areas.index(enc.station_area[s]) for s in enc.stations]))
        self.n_days = enc.n_days

    def forward(self, x):
        loc = self.area(x["area"]) + self.station(x["station"])
        h = self.trunk(torch.cat([loc, self.time(x["day"])], dim=1))
        return torch.cat([head(h) for head in self.heads], dim=1)

    def all_series(self):
        """Predictions for every station over the full day grid, shape (stations, days, targets)."""
        s = torch.arange(len(self.area_of))
        x = {"station": s.repeat_interleave(self.n_days), "area": self.area_of[s].repeat_interleave(self.n_days),
             "day": torch.arange(self.n_days).repeat(len(s))}
        return self(x).view(len(s), self.n_days, len(TARGETS))


def nll(mu, model, y, mask, n):
    var = torch.exp(2 * model.log_sigma) / n + torch.exp(2 * model.log_tau)
    out = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    return (out * mask).sum() / mask.sum()


def roughness(model):
    mu = model.all_series()
    return (mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]).pow(2).mean()


def fit(enc, w, cfg, steps, use=TARGETS, on_checkpoint=None, checkpoints=()):
    """Full-batch Adam on a cosine schedule spanning `cfg['schedule']` steps (same meaning in CV and final fit)."""
    torch.manual_seed(cfg.get("seed", 0))
    model = PrevailingNet(enc, hidden=cfg["hidden"], spacing=cfg["spacing"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["schedule"])
    x, (y, mask, n) = enc.inputs(w), enc.targets(w, use)
    for step in range(1, steps + 1):
        model.train()
        loss = (nll(model(x), model, y, mask, n) + cfg["rough"] * roughness(model)
                + cfg["station_l2"] * model.station.weight.pow(2).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if on_checkpoint and step in checkpoints:
            model.eval()
            on_checkpoint(step, model)
    model.eval()
    return model


def predict(model, enc, keys: pd.DataFrame) -> pd.DataFrame:
    with torch.no_grad():
        v = enc.to_value(model(enc.inputs(keys)))
    return pd.concat([keys[KEY].assign(pollutant=t, pred=v[:, i]) for i, t in enumerate(TARGETS)], ignore_index=True)
