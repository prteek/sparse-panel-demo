"""Streamlit app: compare estimators of a smooth level on sparse, gappy air-quality data.

Reads the tables written by precompute.py: from DATA_DIR if set (or ./outputs if present), otherwise from the
Hugging Face dataset HF_DATASET.
"""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HF_DATASET = "prteek/sparse-panel-demo"
POST_URL = "https://prteek.github.io/regression_room/posts/beijing_air_quality/"
LAMBDA_NAMES = {0.5: "Sparse: ~0.5 hours a day", 1.2: "As in the post: ~1.2 hours a day", 3.0: "Rich: ~3 hours a day"}
LABEL_NOTES = {
    "regular": "Thinned at the global setting, with two 20–60 day gaps per pollutant.",
    "sparse": "Thinned to 40% of the global setting's hours.",
    "very sparse": "As regular, then only 15% of the observed days kept.",
    "long gap": "As regular, but the two gaps are 90–120 days long.",
    "dropped series": "One pollutant removed entirely; each panel shows the dropped one. "
                      "The joint network can rebuild it from the station's other pollutant.",
}
STYLE = {
    "joint": dict(color="#1f77b4", width=2),
    "single-target": dict(color="#9467bd", width=1.5, dash="dash"),
    "±15-day mean": dict(color="#ff7f0e", width=1.2),
}
DAY_TYPES = ["all", "observed", "unobserved", "gap"]
TRUTH_COLOR = "#2ca02c"  # visible on light and dark themes

st.set_page_config(page_title="Sparse panel demo", layout="wide")


@st.cache_data(show_spinner="Loading scenarios…")
def load() -> dict:
    local = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "outputs"))
    if local.exists():
        root = local
    else:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(HF_DATASET, repo_type="dataset", allow_patterns=["*.parquet", "meta.json"]))
    tables = {f.stem: pd.read_parquet(f) for f in root.glob("*.parquet")}
    tables["meta"] = json.loads((root / "meta.json").read_text())
    return tables


def gap_spans(dates: pd.Series):
    d = dates.sort_values()
    block = (d.diff().dt.days != 1).cumsum()
    return d.groupby(block).agg(["min", "max"]).itertuples(index=False)


def figure(t: dict, lam: float, rows: pd.DataFrame) -> go.Figure:
    titles = [f"{r.station} · {r.pollutant} ({r.area})" for r in rows.itertuples()]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.06)
    readings = t["readings"][t["readings"]["lam"] == lam]
    preds = t["predictions"][t["predictions"]["lam"] == lam]
    for i, r in enumerate(rows.itertuples(), 1):
        sel = lambda d: d[(d["station"] == r.station) & (d["pollutant"] == r.pollutant)]
        first = i == 1
        s = sel(readings)
        fig.add_trace(go.Scatter(x=s["date"], y=s["value"], mode="markers", name="sparse readings",
                                 marker=dict(size=3 + 1.5 * s["n"], color="rgba(128,128,128,0.45)"),
                                 legendgroup="readings", showlegend=first), row=i, col=1)
        tr = sel(t["truth"])
        fig.add_trace(go.Scatter(x=tr["date"], y=tr["value"], mode="lines", name="31-day MA (truth)",
                                 line=dict(color=TRUTH_COLOR, width=2.5), legendgroup="truth", showlegend=first),
                      row=i, col=1)
        for name, style in STYLE.items():
            p = sel(preds[preds["estimator"] == name])
            fig.add_trace(go.Scatter(x=p["date"], y=p["pred"], mode="lines", name=name, line=style,
                                     legendgroup=name, showlegend=first), row=i, col=1)
        # after the traces: plotly skips shapes on subplots that have no traces yet
        for a, b in gap_spans(sel(t["gaps"])["date"]):
            fig.add_vrect(x0=a, x1=b, fillcolor="gold", opacity=0.25, line_width=0, layer="below", row=i, col=1)
        fig.update_yaxes(type="log", title_text="µg/m³", row=i, col=1)
    height, top = 260 * len(rows) + 110, 100
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), hovermode="x unified",
                      # legend sits in the top margin, clear of the first subplot title
                      legend=dict(orientation="h", y=1 + 45 / (height - top - 10), yanchor="bottom", x=0))
    return fig


def error_table(m: pd.DataFrame) -> pd.DataFrame:
    tab = m.pivot_table(index="estimator", columns="day_type", values="error")
    days = m.groupby("day_type")["days"].max()
    cols = [c for c in DAY_TYPES if c in tab]
    tab = tab[cols].reindex(list(STYLE))
    tab.columns = [f"{c} ({days[c]:,} days)" for c in cols]
    return tab.round(3)


t = load()
labels = t["labels"]

st.title("Smooth levels from sparse, gappy data")
st.markdown(
    "Hourly PM2.5 and PM10 from 12 Beijing monitoring stations are thinned to a few readings a day, with gaps, "
    "and three estimators try to recover the slow underlying level: a **joint network** that models both "
    "pollutants together, the same network fitted to **one pollutant at a time**, and a **±15-day mean** of the "
    f"readings. The truth is the 31-day moving average of the full data. Method and discussion: [blog post]({POST_URL})."
)

with st.sidebar:
    st.header("Scenario")
    lam = st.radio("Readings kept (Poisson mean hours per station-day)", t["meta"]["lambdas"],
                   index=t["meta"]["lambdas"].index(1.2), format_func=lambda l: LAMBDA_NAMES.get(l, str(l)))
    st.header("View")
    label = st.selectbox("Station group", list(LABEL_NOTES))
    if label == "dropped series":
        st.caption("Each station shows the pollutant that was dropped.")
        pollutant = "dropped"
    else:
        pollutant = st.radio("Pollutant", ["PM2.5", "PM10"], horizontal=True)

group = labels[labels["label"] == label]
rows = group.assign(pollutant=group["dropped"] if label == "dropped series" else pollutant)

st.subheader(f"{label.capitalize()} stations")
st.caption(LABEL_NOTES[label] + " Shaded: gap blocks.")
st.plotly_chart(figure(t, lam, rows), width="stretch")

m = t["metrics"][(t["metrics"]["label"] == label) & (t["metrics"]["pollutant"] == pollutant)]
left, right = st.columns([3, 2])
with left:
    st.markdown("**Error against the 31-day MA** (mean absolute log error; 0.15 ≈ 15%)")
    st.dataframe(error_table(m[m["lam"] == lam]), width="stretch")
    st.caption("observed: days with a reading · unobserved: no reading, outside the gaps · gap: inside a gap block")
with right:
    st.markdown("**All days, by thinning level**")
    trend = m[m["day_type"] == "all"].pivot_table(index="lam", columns="estimator", values="error")[list(STYLE)]
    fig = go.Figure([go.Scatter(x=trend.index, y=trend[c], mode="lines+markers", name=c,
                                line={k: v for k, v in STYLE[c].items() if k != "width"}) for c in trend])
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="mean hours kept per day",
                      yaxis_title="error", legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(fig, width="stretch")

st.caption("Data: Beijing Multi-Site Air Quality, UCI Machine Learning Repository (CC BY 4.0). "
           "Labels are assigned once with a fixed seed: each degraded group has one urban and one suburban station.")
