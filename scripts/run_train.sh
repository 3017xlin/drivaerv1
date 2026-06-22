#!/usr/bin/env bash
# Training launcher for 1-8 GPU single node.
set -euo pipefail

WORLD="${WORLD:-4}"
B="${B:-1}"
CACHE_DIR="${CACHE_DIR:-cache_16}"
RUN_DIR="${RUN_DIR:-runs/$(date +%Y%m%d_%H%M%S)}"

torchrun --nproc-per-node "$WORLD" train.py \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$B" \
    --run_dir "$RUN_DIR" \
    "$@"
