#!/usr/bin/env bash
# Publish to Hugging Face (needs `hf auth login` as prteek):
#   dataset  outputs/ tables from huggingface/dataset/precompute.py, with huggingface/dataset/README.md as the card
#   space    huggingface/space/ (the Gradio app reads the dataset at runtime)
#   model    sparse_panel/model.py with huggingface/model/ (code and training recipe, no weights)
# Usage: huggingface/publish.sh [message] [dataset|space|model ...]   (default: all three)
set -euo pipefail
cd "$(dirname "$0")/.."
msg="${1:-Update}"
shift || true
if [ $# -gt 0 ]; then targets=("$@"); else targets=(dataset space model); fi
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

for t in "${targets[@]}"; do
  mkdir "$stage/$t"
  case "$t" in
    dataset)
      cp outputs/*.parquet outputs/meta.json "$stage/dataset/"
      cp huggingface/dataset/README.md "$stage/dataset/README.md"
      uv run hf upload prteek/sparse-panel-demo "$stage/dataset" . --repo-type dataset --commit-message "$msg" ;;
    space)
      cp huggingface/space/app.py huggingface/space/README.md huggingface/space/requirements.txt "$stage/space/"
      uv run hf upload prteek/sparse-panel-demo "$stage/space" . --repo-type space --commit-message "$msg" ;;
    model)
      cp sparse_panel/model.py huggingface/model/example.py huggingface/model/README.md "$stage/model/"
      uv run hf upload prteek/sparse-panel-demo "$stage/model" . --repo-type model --commit-message "$msg" ;;
    *)
      echo "unknown target: $t" >&2; exit 1 ;;
  esac
done
