"""Training entry point.

    torchrun --nproc-per-node 4 train.py --cache_dir cache_16 \
                                          --batch_size 1 --num_epochs 400
"""
import argparse
import os
import os.path as osp
import time

import yaml

from training.loop import train


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
    ap.add_argument('--run_dir', default=None,
                    help='Output directory; default runs/<timestamp>.')
    args = ap.parse_args()

    cfg = _apply_overrides(_load_cfg(args.config), args)
    run_dir = args.run_dir or osp.join(cfg['output_base'],
                                       time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    train(cfg, run_dir)


if __name__ == '__main__':
    main()
