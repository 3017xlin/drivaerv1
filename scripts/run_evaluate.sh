#!/usr/bin/env bash
# Post-training: curve + test eval + viz + report.
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-cache_16}"
RUN_DIR="${RUN_DIR:?must set RUN_DIR}"

# Curve uses DDP if WORLD > 1; test eval forces single GPU regardless.
WORLD="${WORLD:-1}"
if [[ "$WORLD" -gt 1 ]]; then
    torchrun --nproc-per-node "$WORLD" evaluate.py \
        --cache_dir "$CACHE_DIR" --run_dir "$RUN_DIR" "$@"
else
    python evaluate.py --cache_dir "$CACHE_DIR" --run_dir "$RUN_DIR" "$@"
fi
