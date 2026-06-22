"""Post-training entry point: curve + test eval + viz + report.

  python evaluate.py --cache_dir cache_16 --run_dir runs/<timestamp>

Stages:
  1. curve  — DDP forward across all saved checkpoints; produce
              train_val_curve.{json,png}; delete each checkpoint after use.
  2. test   — single-GPU streaming over 50 test cases; produce
              per_case metrics + Cd / Cl.
  3. viz    — cdcl_scatter, per_case_error_hist, vol_slice_*, surf_*.
  4. report — eval_summary.json + tables.tex + terminal print.

Skip any stage with --skip_<stage>.
"""
import argparse
import json
import os.path as osp
import time

import torch
import yaml

from evaluation.curve import run_curve
from models import DrivAer3DModel
from evaluation.test_eval import run_test_eval
from evaluation.viz import (cdcl_scatter, per_case_error_hist,
                             surface_field_views, vol_slice_velocity_pressure)
from reporting.summary import build_eval_summary, write_eval_summary
from reporting.tables import print_summary, render_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--skip_curve', action='store_true')
    ap.add_argument('--skip_test', action='store_true')
    ap.add_argument('--skip_viz', action='store_true')
    ap.add_argument('--skip_report', action='store_true')
    ap.add_argument('--delete_checkpoints', action='store_true',
                    help='Delete checkpoints after curve evaluation')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['data']['cache_dir'] = args.cache_dir

    timing: dict[str, float] = {}

    t0 = time.time()
    if not args.skip_curve:
        run_curve(cfg, args.run_dir,
                  delete_checkpoints=args.delete_checkpoints)
        timing['curve_total_sec'] = time.time() - t0

    if not args.skip_test:
        t1 = time.time()
        per_case = run_test_eval(cfg, args.run_dir)
        timing['eval_total_sec'] = time.time() - t1
        timing['inference_per_case_sec'] = (
            timing['eval_total_sec'] / max(len(per_case), 1))
    else:
        per_case = {}

    if per_case and not args.skip_viz:
        cdcl_scatter(per_case, osp.join(args.run_dir, 'cdcl_scatter.png'))
        per_case_error_hist(per_case,
                            osp.join(args.run_dir, 'per_case_error_hist.png'))
        vol_slice_velocity_pressure(per_case, args.run_dir,
                                    y_tol=float(cfg['evaluation'][
                                        'vol_slice_y_tolerance']),
                                    pct=tuple(cfg['evaluation'][
                                        'vol_slice_colorbar_percentiles']))
        try:
            from dataset.loaders import load_coef_norm
            coef_norm = load_coef_norm(args.cache_dir)
            surface_field_views(per_case, args.run_dir,
                                coef_norm=coef_norm)
        except Exception as e:
            print(f'[viz] surface_field_views failed: {e}')

    if per_case and not args.skip_report:
        model_info = {
            'total_params': sum(p.numel() for p in
                                DrivAer3DModel(cfg).parameters()),
            'gpu_peak_mem_gib': (torch.cuda.max_memory_allocated() / 1024**3
                                  if torch.cuda.is_available() else 0.0),
        }
        summary = build_eval_summary(per_case, timing, model_info)
        write_eval_summary(summary, osp.join(args.run_dir,
                                              'eval_summary.json'))
        render_tables(summary, cfg, osp.join(args.run_dir, 'tables.tex'))
        print_summary(summary)


if __name__ == '__main__':
    main()
