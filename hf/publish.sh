#!/usr/bin/env bash
# Publish the precomputed outputs as the HF dataset, the app as the HF Space and the model code as the HF model repo (needs `hf auth login` as prteek).
set -euo pipefail
cd "$(dirname "$0")/.."
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir "$stage/dataset" "$stage/space" "$stage/model"
cp outputs/*.parquet outputs/meta.json "$stage/dataset/"
cp hf/dataset_card.md "$stage/dataset/README.md"
uv run hf upload prteek/sparse-panel-demo "$stage/dataset" . --repo-type dataset --commit-message "${1:-Update scenarios}"

cp app.py hf/space/README.md hf/space/requirements.txt "$stage/space/"
uv run hf upload prteek/sparse-panel-demo "$stage/space" . --repo-type space --commit-message "${1:-Update app}"

# Model repo: code and training recipe only (fitted weights are tied to one panel)
cp sparse_panel/model.py hf/model/example.py hf/model/README.md "$stage/model/"
uv run hf upload prteek/sparse-panel-demo "$stage/model" . --repo-type model --commit-message "${1:-Update model code}"
