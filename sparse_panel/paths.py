"""Repository folders shared by the blog post, the CV scripts and the Hugging Face artefacts."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"         # raw UCI download and derived parquet files (not in git)
RESULTS = REPO / "results"   # small, tracked outputs of the blog's CV runs that the blog post reads
OUTPUTS = REPO / "outputs"   # large or regenerable outputs: demo tables, fitted weights (not in git)
