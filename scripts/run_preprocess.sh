#!/usr/bin/env bash
# Preprocess on a 60-core node.
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-cache_16}"
STEP1_DIR="${STEP1_DIR:-$HOME/scratch/drivaerml_pt}"
MAX_WORKERS="${MAX_WORKERS:-60}"

python make_manifest.py --cache_dir "$CACHE_DIR"
python preprocess.py --step1_dir "$STEP1_DIR" \
                     --cache_dir "$CACHE_DIR" \
                     --max_workers "$MAX_WORKERS"
