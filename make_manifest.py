"""Write cache_16/manifest.json from the hard-coded split.

Run once before preprocess.py. Idempotent.
"""
import argparse
import json
import os.path as osp

from dataset.split_ids import (HIDDEN_VAL_IDS, TEST_IDS, TRAIN_IDS, VAL_IDS,
                               build_train_eval_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    manifest = {
        'train_ids':      TRAIN_IDS,
        'val_ids':        VAL_IDS,
        'test_ids':       TEST_IDS,
        'train_eval_ids': build_train_eval_ids(seed=args.seed),
        'hidden_val_ids': HIDDEN_VAL_IDS,
    }
    out = osp.join(args.cache_dir, 'manifest.json')
    with open(out, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote {out} '
          f'(train={len(TRAIN_IDS)}, val={len(VAL_IDS)}, '
          f'test={len(TEST_IDS)}, train_eval={len(manifest["train_eval_ids"])})')


if __name__ == '__main__':
    main()
