#!/bin/bash
set -euo pipefail

# Bootstrap Mentat stack-trace training on top of nanochat's existing pipeline.
#
# Example:
#   bash runs/mentat_stack.sh
#
# Override knobs:
#   NPROC_PER_NODE=8
#   NANOCHAT_BASE_DIR=/mnt/nvme/nanochat
#   NANOCHAT_DATA_DIR=/mnt/nvme/nanochat/base_data_mentat_stack

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$BASE_DIR/base_data_mentat_stack}"
NPROC="${NPROC_PER_NODE:-8}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "No python interpreter found. Run 'uv sync' first or set up a project venv." >&2
  exit 1
fi

if [[ -x ".venv/bin/torchrun" ]]; then
  TORCHRUN_BIN=".venv/bin/torchrun"
elif command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_BIN="$(command -v torchrun)"
else
  echo "No torchrun found. Run 'uv sync' first or install PyTorch into the active environment." >&2
  exit 1
fi

echo "Mentat dataset dir: $NANOCHAT_DATA_DIR"
echo "Using $NPROC processes for training"
echo "Python: $PYTHON_BIN"
echo "Torchrun: $TORCHRUN_BIN"

"$PYTHON_BIN" -m scripts.mentat_data \
  --output-dir "$NANOCHAT_DATA_DIR" \
  --train-docs 1000000 \
  --val-docs 10000 \
  --target-chars-per-shard 50000000 \
  --overwrite

"$PYTHON_BIN" -m scripts.tok_train \
  --max-chars 500000000 \
  --doc-cap 20000 \
  --vocab-size 16384

OMP_NUM_THREADS=1 "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
  --run="mentat-stack-d12" \
  --model-tag="mentat-stack-d12" \
  --depth=12 \
  --max-seq-len=512 \
  --window-pattern=L \
  --device-batch-size=32 \
  --total-batch-size=262144 \
  --num-iterations=20000 \
  --target-param-data-ratio=-1 \
  --eval-every=100 \
  --eval-tokens=1048576 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=1000
