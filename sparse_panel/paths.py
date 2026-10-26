"""Repository folders shared by the paper, the CV scripts and the Hugging Face artefacts."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"         # raw UCI download and derived parquet files (not in git)
RESULTS = REPO / "results"   # small, tracked outputs of the paper's CV runs that the paper reads
OUTPUTS = REPO / "outputs"   # large or regenerable outputs: demo tables, fitted weights (not in git)
