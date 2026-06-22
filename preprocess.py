"""Entry point for preprocess Step 2.

  python preprocess.py --step1_dir ~/scratch/drivaerml_pt --cache_dir cache_16

Requires manifest.json to exist in cache_dir (run make_manifest.py first).
"""
import argparse
import json
import os.path as osp
import sys

from preprocess.pipeline import run_preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step1_dir', required=True,
                    help='Directory containing run_<id>.pt from step 1.')
    ap.add_argument('--cache_dir', required=True,
                    help='Output cache directory (will write coef_norm.pt, '
                         'case_<id>.pt files, and test/).')
    ap.add_argument('--max_workers', type=int, default=60)
    ap.add_argument('--n_query', type=int, default=500_000)
    ap.add_argument('--surface_query_ratio', type=float, default=0.2)
    ap.add_argument('--surface_area_alpha', type=float, default=1.0)
    ap.add_argument('--scratch_dir', type=str, default=None)
    args = ap.parse_args()

    manifest_path = osp.join(args.cache_dir, 'manifest.json')
    if not osp.exists(manifest_path):
        sys.exit(f'manifest.json not found in {args.cache_dir}; '
                 f'run `python make_manifest.py --cache_dir {args.cache_dir}` first.')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    n_query_vol = int(args.n_query * (1.0 - args.surface_query_ratio))
    run_preprocess(step1_dir=args.step1_dir, cache_dir=args.cache_dir,
                   manifest=manifest, max_workers=args.max_workers,
                   scratch_dir=args.scratch_dir,
                   n_query=args.n_query, n_query_vol=n_query_vol,
                   surface_area_alpha=args.surface_area_alpha)


if __name__ == '__main__':
    main()
