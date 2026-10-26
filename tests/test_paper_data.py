"""The paper's sparse data must be reproduced exactly: every number in the paper depends on it."""
import numpy as np
import pandas as pd
import pytest

from sparse_panel.data import CSV_DIR, PAPER_STATION_ORDER, SEED, daily, load_hourly, thin_uniform
from sparse_panel.paths import DATA

pytestmark = pytest.mark.skipif(not (DATA / CSV_DIR).exists(), reason="UCI data not downloaded")


def test_paper_sparse_data_is_reproduced():
    hourly = load_hourly(DATA / CSV_DIR, station_order=PAPER_STATION_ORDER)
    sparse, gaps = thin_uniform(hourly, np.random.default_rng(SEED))
    assert len(daily(hourly)) == 34955
    assert len(sparse) == 23243
    assert len(gaps) == 1978
    # fingerprint of the kept readings, so a different draw with the same row count still fails
    assert sparse["n"].sum() == 39859
    assert np.isclose(np.log(sparse["value"]).sum(), 94993.9, rtol=1e-6)


def test_sorted_order_gives_different_data():
    """Guards the reason the order is pinned: stacking alphabetically draws a different sample."""
    hourly = load_hourly(DATA / CSV_DIR)
    sparse, _ = thin_uniform(hourly, np.random.default_rng(SEED))
    assert len(sparse) != 23243 or sparse["n"].sum() != 39859
