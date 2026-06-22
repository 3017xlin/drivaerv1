"""One-shot entry: train → curve → test → viz → report.

    torchrun --nproc-per-node 4 run.py --cache_dir cache_16 --batch_size 2

All stages run in the same torchrun session. Curve deletes checkpoints
after evaluation to save disk. Use --skip_curve / --skip_test / --skip_viz
/ --skip_report to skip stages.
"""
import argparse
import os
import os.path as osp
import time

import torch
import yaml

from training.loop import train
from evaluation.curve import run_curve
from evaluation.test_eval import run_test_eval
from evaluation.viz import (cdcl_scatter, per_case_error_hist,
                             surface_field_views, vol_slice_velocity_pressure)
from models import DrivAer3DModel
from reporting.summary import build_eval_summary, write_eval_summary
from reporting.tables import print_summary, render_tables
from training.ddp import init_ddp, cleanup_ddp


def _load_cfg(path: str = 'config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.cache_dir is not None:
        cfg['data']['cache_dir'] = args.cache_dir
    if args.num_epochs is not None:
        cfg['training']['num_epochs'] = int(args.num_epochs)
    if args.batch_size is not None:
        cfg['training']['batch_size'] = int(args.batch_size)
    if args.num_workers is not None:
        cfg['training']['num_workers'] = int(args.num_workers)
    if args.lr is not None:
        cfg['training']['lr'] = float(args.lr)
    if args.log_training_nut:
        cfg['log_training']['nut'] = True
    if args.log_training_vorticity:
        cfg['log_training']['vorticity'] = True
    if args.N_query is not None:
        cfg['sampling']['N_query'] = int(args.N_query)
    if args.surface_query_ratio is not None:
        cfg['sampling']['surface_query_ratio'] = float(
            args.surface_query_ratio)
    if args.surface_area_alpha is not None:
        cfg['sampling']['surface_area_alpha'] = float(args.surface_area_alpha)
    if args.seed is not None:
        cfg['seed'] = int(args.seed)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--cache_dir', default=None)
    ap.add_argument('--run_name', default=None,
                    help='Run name; default is timestamp.')
    ap.add_argument('--num_epochs', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--log_training_nut', action='store_true')
    ap.add_argument('--log_training_vorticity', action='store_true')
    ap.add_argument('--N_query', type=int, default=None)
    ap.add_argument('--surface_query_ratio', type=float, default=None)
    ap.add_argument('--surface_area_alpha', type=float, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--skip_curve', action='store_true')
    ap.add_argument('--skip_test', action='store_true')
    ap.add_argument('--skip_viz', action='store_true')
    ap.add_argument('--skip_report', action='store_true')
    ap.add_argument('--delete_checkpoints', action='store_true', default=True,
                    help='Delete checkpoints after curve (default: true)')
    ap.add_argument('--keep_checkpoints', action='store_true')
    args = ap.parse_args()

    cfg = _apply_overrides(_load_cfg(args.config), args)

    name = args.run_name or time.strftime('%Y%m%d_%H%M%S')
    run_dir = osp.join(cfg['output_base'], name)
    os.makedirs(run_dir, exist_ok=True)

    rank = int(os.environ.get('RANK', 0))
    timing: dict[str, float] = {}

    # ---- Stage 1: Train ----
    if rank == 0:
        print(f'=== TRAIN === {name}', flush=True)
    t0 = time.time()
    train(cfg, run_dir)
    timing['train_total_sec'] = time.time() - t0
    if rank == 0:
        print(f'=== TRAIN DONE === {timing["train_total_sec"]:.0f}s', flush=True)

    # ---- Stage 2: Curve ----
    if not args.skip_curve:
        if rank == 0:
            print('=== CURVE ===', flush=True)
        t1 = time.time()
        delete_ckpt = args.delete_checkpoints and not args.keep_checkpoints
        run_curve(cfg, run_dir, delete_checkpoints=delete_ckpt)
        timing['curve_total_sec'] = time.time() - t1
        if rank == 0:
            print(f'=== CURVE DONE === {timing["curve_total_sec"]:.0f}s', flush=True)

    # ---- Stage 3: Test eval (rank 0 only in run_test_eval) ----
    per_case = {}
    if not args.skip_test:
        if rank == 0:
            print('=== TEST EVAL ===', flush=True)
        t2 = time.time()
        per_case = run_test_eval(cfg, run_dir)
        timing['eval_total_sec'] = time.time() - t2
        timing['inference_per_case_sec'] = (
            timing['eval_total_sec'] / max(len(per_case), 1))
        if rank == 0:
            print(f'=== TEST EVAL DONE === {timing["eval_total_sec"]:.0f}s', flush=True)

    # ---- Stage 4: Viz + Report (rank 0 only) ----
    if rank == 0:
        cache_dir = cfg['data']['cache_dir']
        if per_case and not args.skip_viz:
            print('=== VIZ ===', flush=True)
            cdcl_scatter(per_case, osp.join(run_dir, 'cdcl_scatter.png'))
            per_case_error_hist(per_case,
                                osp.join(run_dir, 'per_case_error_hist.png'))
            vol_slice_velocity_pressure(
                per_case, run_dir,
                y_tol=float(cfg['evaluation']['vol_slice_y_tolerance']),
                pct=tuple(cfg['evaluation']['vol_slice_colorbar_percentiles']))
            try:
                from dataset.loaders import load_coef_norm
                coef_norm = load_coef_norm(cache_dir)
                surface_field_views(per_case, run_dir, coef_norm=coef_norm)
            except Exception as e:
                print(f'[viz] surface_field_views failed: {e}')

        if per_case and not args.skip_report:
            print('=== REPORT ===', flush=True)
            model_info = {
                'total_params': sum(
                    p.numel() for p in DrivAer3DModel(cfg).parameters()),
                'gpu_peak_mem_gib': (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if torch.cuda.is_available() else 0.0),
            }
            summary = build_eval_summary(per_case, timing, model_info)
            write_eval_summary(summary,
                               osp.join(run_dir, 'eval_summary.json'))
            render_tables(summary, cfg, osp.join(run_dir, 'tables.tex'))
            print_summary(summary)

        print(f'=== ALL DONE === {name}', flush=True)

    cleanup_ddp()


if __name__ == '__main__':
    main()
