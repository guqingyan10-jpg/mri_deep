# Full from random initialization on BraTS2020 and UCSF-BMSR

This experiment changes the Full model's initialization only. It retains LHFC
concat, ABS weight 0.1, AR-MSC V2, 24 base channels, each dataset's existing
preprocessing and split, Adam 5e-4, batch size 1, accumulation 4,
ReduceLROnPlateau patience 2, and validation-loss early stopping with patience
25/min_delta 1e-4. Keep the existing environment; do not reinstall PyTorch.

Use seed 55 first, matching the single-run comparison. Random initialization
does not guarantee a higher test score. This is an initialization ablation,
not a replication of nnU-Net's independent 1000-epoch training recipe.

## Pull and check the data locations

```bash
cd /root/autodl-tmp/mri_deep
git switch master
git pull --ff-only origin master
command -v flock
test -f tumourCSV.csv
test -f /root/autodl-tmp/ucsf_bmsr_prepared/manifest.csv
```

BraTS uses the **same** `tumourCSV.csv` as the prior models; preserve row order
and its 257/74/37 assignment. Its path column must still point to the original
MRI files. UCSF uses the **same** prepared-data manifest and patient split as
the prior Baseline/Full runs. Pass the actual existing paths if different.
These inputs are local study files, not model weights to download from GitHub.

## BraTS on GPU 0

First check GPU availability. A still-running official job must finish before
reusing its GPU. Start the command once; the lock prevents duplicate launches
using this same output directory.

```bash
mkdir -p /root/autodl-tmp/afbms_full_from_scratch/brats/seed55

nohup flock -n /root/autodl-tmp/afbms_full_from_scratch/brats/seed55/.train.lock \
  env CUDA_VISIBLE_DEVICES=0 python -u scripts/train_hf_concat_boundary.py \
  --from-scratch --seed 55 --fusion concat \
  --boundary_weight 0.1 --multiscale_context_v2 \
  --epochs 200 --lr 0.0005 \
  --csv /root/autodl-tmp/mri_deep/tumourCSV.csv \
  --checkpoint_dir /root/autodl-tmp/afbms_full_from_scratch/brats/seed55 \
  >> /root/autodl-tmp/afbms_full_from_scratch/brats/seed55/run.log 2>&1 &
```

## UCSF on GPU 1

`--models full --from-scratch` neither trains nor requires a paired Baseline.
It retains the existing binary Full model, prepared images, and grouped split.

```bash
mkdir -p /root/autodl-tmp/afbms_full_from_scratch/ucsf/seed55

nohup flock -n /root/autodl-tmp/afbms_full_from_scratch/ucsf/seed55/.train.lock \
  env CUDA_VISIBLE_DEVICES=1 python -u scripts/train_ucsf_baseline_full.py \
  --models full --from-scratch --seed 55 --epochs 200 --lr 0.0005 \
  --manifest /root/autodl-tmp/ucsf_bmsr_prepared/manifest.csv \
  --output-root /root/autodl-tmp/afbms_full_from_scratch/ucsf \
  >> /root/autodl-tmp/afbms_full_from_scratch/ucsf/seed55/run.log 2>&1 &
```

UCSF stores weights in `ucsf/seed55/full`; BraTS stores them directly in
`brats/seed55`. These are separate from all earlier warm-start runs.

## Verify initialization and stopping

```bash
tail -n 60 -f /root/autodl-tmp/afbms_full_from_scratch/brats/seed55/run.log
tail -n 60 -f /root/autodl-tmp/afbms_full_from_scratch/ucsf/seed55/run.log
```

Use separate terminals for the two `tail` commands. Ctrl+C exits the viewer.
The logs must report random/from-scratch initialization, with no Baseline
weight loading. `initialization_protocol.json` in each model directory records
the seed, source-manifest hash, model settings and early-stopping rule. A
scratch run refuses unlabelled existing weights, changed settings, or reuse by
the warm-start mode; choose a new directory instead of deleting old results.

The original training loop is deliberately unchanged: `--epochs 200` retains
its existing epoch-index convention (fresh logs run 1 through at most 199).
Validation loss selects `best_model_N.pth`; early stopping may finish sooner.
Neither Full entry point automatically runs the official nnU-Net evaluator.
Evaluate the selected best Full checkpoint with the existing Full model class
and common scoring rules after training; never load it into an nnU-Net model.

The legacy Trainer resumes weights and history from its own output directory,
but it does not restore optimizer/scheduler/RNG state exactly. An uninterrupted
scratch run is the cleanest initialization comparison. If interrupted, record
that fact instead of claiming exact continuation. For a later independent seed,
change both the seed argument and every corresponding output/log/lock path.
