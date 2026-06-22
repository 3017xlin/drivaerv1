# drivaer3d

DrivAerML 3D surrogate per v4 plan. Three entry points run in sequence:

```bash
# 1. Preprocess (60-core node, ~45 min)
python preprocess.py --step1_dir ~/scratch/drivaerml_pt \
                     --cache_dir  cache_16

# 2. Train (1-8 GPU node, ~3.5 h DDP world=4)
torchrun --nproc-per-node 4 train.py --cache_dir cache_16 \
                                     --num_epochs 400 \
                                     --batch_size 1

# 3. Post-train (curve + test eval + viz + report; ~15 min)
python evaluate.py --cache_dir cache_16 \
                   --run_dir runs/<timestamp>
```

Full spec: `../v4计划书.txt`. Validation checklist: chapter §18.

## Layout

```
drivaer3d/
├── config.yaml                  default hyperparameters
├── preprocess.py / train.py / evaluate.py
├── make_manifest.py             generate manifest.json (train/val/test/train_eval IDs)
├── dataset/                     pinned-RAM loaders + async prefetcher
├── preprocess/                  Phase 0-11 (KD-tree, geometry, leaf stats, Welford, ...)
├── models/                      Encoder cross-attn + 12-layer ViT (dense SwiGLU) + Decoder
├── training/                    DDP loop, transient1/2 worker, SWA, checkpointing
├── evaluation/                  Curve / test inference / metrics / Cd-Cl R² / viz
├── reporting/                   eval_summary.json + tables.tex
├── utils/                       ResourceMonitor, seeding, memory tracking
└── tests/                       KD-tree invariants, Welford bit-exact
```

## Hard requirements (do not silently relax)

- Volume head dim 0 is **raw p_v** (not Cpt).
- ViT FFN is **dense SwiGLU on all 12 layers** (no MoE / router / expert).
- Only **two normalization sets**: coordinate (norm_p5/p95) + field z-score.
  No independent RoPE percentile.
- Test target stored as **raw fp32**; train/val/train_eval as bf16 z-scored.
- KD-tree leaf assignment comes from the **build-time interval**, never
  from a descent function (descent is allowed only for test's non-keep
  87.5% points).
- Each rank pins **only the 400 train cases** (~356 GB); val and test
  PT remain on disk and are loaded on demand.
