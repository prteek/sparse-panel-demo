#!/usr/bin/env bash
# Publish the precomputed outputs as the HF dataset and the app as the HF Space (needs `hf auth login` as prteek).
set -euo pipefail
cd "$(dirname "$0")/.."
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir "$stage/dataset" "$stage/space"
cp outputs/*.parquet outputs/meta.json "$stage/dataset/"
cp hf/dataset_card.md "$stage/dataset/README.md"
uv run hf upload prteek/sparse-panel-demo "$stage/dataset" . --repo-type dataset --commit-message "${1:-Update scenarios}"

cp app.py hf/space/README.md hf/space/requirements.txt hf/space/Dockerfile "$stage/space/"
uv run hf upload prteek/sparse-panel-demo "$stage/space" . --repo-type space --commit-message "${1:-Update app}"
