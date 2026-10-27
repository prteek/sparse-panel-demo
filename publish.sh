#!/usr/bin/env bash
# Publish to Hugging Face (locally: `hf auth login` as prteek; in CI: HF_TOKEN):
#   dataset       outputs/ tables from dataset/precompute.py, with dataset/README.md as the card
#   dataset-card  only the dataset card (CI: the tables are rebuilt and published by hand)
#   space         space/ (the Gradio app reads the dataset at runtime)
#   model         sparse_panel/model.py with model/ (code and training recipe, no weights)
# Usage: ./publish.sh [message] [dataset|dataset-card|space|model ...]   (default: dataset space model)
# Set HF_CLI=hf to use an installed hf CLI instead of the project environment (as CI does).
set -euo pipefail
cd "$(dirname "$0")"
msg="${1:-Update}"
shift || true
if [ $# -gt 0 ]; then targets=("$@"); else targets=(dataset space model); fi
read -r -a hf <<< "${HF_CLI:-uv run hf}"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

for t in "${targets[@]}"; do
  mkdir "$stage/$t"
  case "$t" in
    dataset)
      cp outputs/*.parquet outputs/meta.json "$stage/dataset/"
      cp dataset/README.md "$stage/dataset/README.md"
      "${hf[@]}" upload prteek/sparse-panel-demo "$stage/dataset" . --repo-type dataset --commit-message "$msg" ;;
    dataset-card)
      "${hf[@]}" upload prteek/sparse-panel-demo dataset/README.md README.md --repo-type dataset \
        --commit-message "$msg" ;;
    space)
      cp space/app.py space/README.md space/requirements.txt "$stage/space/"
      "${hf[@]}" upload prteek/sparse-panel-demo "$stage/space" . --repo-type space --commit-message "$msg" ;;
    model)
      cp sparse_panel/model.py model/example.py model/README.md "$stage/model/"
      "${hf[@]}" upload prteek/sparse-panel-demo "$stage/model" . --repo-type model --commit-message "$msg" ;;
    *)
      echo "unknown target: $t" >&2; exit 1 ;;
  esac
done
