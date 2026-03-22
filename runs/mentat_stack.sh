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

echo "Mentat dataset dir: $NANOCHAT_DATA_DIR"
echo "Using $NPROC processes for training"

python -m scripts.mentat_data \
  --output-dir "$NANOCHAT_DATA_DIR" \
  --train-docs 1000000 \
  --val-docs 10000 \
  --target-chars-per-shard 50000000 \
  --overwrite

python -m scripts.tok_train \
  --max-chars 500000000 \
  --doc-cap 20000 \
  --vocab-size 16384

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
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
