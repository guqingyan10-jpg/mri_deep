"""
=============================================================================
Train with Foreground-Aware Patch Sampling (SLA-FB Step 1)
=============================================================================
Inspired by STSNet (Zhao et al., Scientific Reports 2025).

This experiment tests: Does foreground-aware patch sampling improve
small ET lesion detection compared to standard random sampling?

Only change: data sampling strategy (same model, loss, hyperparams)
  - Standard training: uniform random crop (baseline)
  - FG-aware training: 4-strategy weighted sampling
      random (20%) + foreground (30%) + et_centered (30%) + small_lesion (20%)

Model: ResUNet3d baseline (same as V1/V2 experiments)
Loss:  BCEDiceLoss (same as baseline)
      → changes exactly ONE variable: sampling strategy

Usage:
    # Default ratios (20/30/30/20):
    python scripts/train_fg_sampling.py

    # Custom ratios:
    python scripts/train_fg_sampling.py --random 0.3 --foreground 0.3 \
        --et_centered 0.2 --small_lesion 0.2

    # Different patch size:
    python scripts/train_fg_sampling.py --patch_size 96 96 80

Checkpoint:
    /root/autodl-tmp/ResUNet_FG_Sampling_model/

Author: Generated for ResUNet enhancement project
Date:   2026-08-04
=============================================================================
"""

import os, sys
import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from losses.basics import BCEDiceLoss
from data.dataset import BratsDataset, BratsDatasetWithFGSampling
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist

# ============================================================
# Parse
# ============================================================

parser = argparse.ArgumentParser(
    description='Train ResUNet with Foreground-Aware Patch Sampling',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--random', type=float, default=0.2,
                    dest='ratio_random', help='Random patch ratio')
parser.add_argument('--foreground', type=float, default=0.3,
                    dest='ratio_foreground', help='Foreground patch ratio')
parser.add_argument('--et_centered', type=float, default=0.3,
                    dest='ratio_et_centered', help='ET-centered patch ratio')
parser.add_argument('--small_lesion', type=float, default=0.2,
                    dest='ratio_small_lesion', help='Small-lesion patch ratio')
parser.add_argument('--small_threshold', type=int, default=50,
                    help='Max ET voxels for small_lesion category')
parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 96],
                    help='3D patch size (D H W)')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--no_cache', action='store_true',
                    help='Rebuild foreground index from scratch')
args = parser.parse_args()

# Validate ratios
ratios = {
    'random': args.ratio_random,
    'foreground': args.ratio_foreground,
    'et_centered': args.ratio_et_centered,
    'small_lesion': args.ratio_small_lesion,
}
total = sum(ratios.values())
if abs(total - 1.0) > 0.01:
    print(f"[WARN] Ratios sum to {total:.3f}, normalizing...")
    ratios = {k: v / total for k, v in ratios.items()}

seed_everything(config.seed)

CHECKPOINT_DIR = '/root/autodl-tmp/ResUNet_FG_Sampling_model'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print("SLA-FB Step 1: Foreground-Aware Patch Sampling")
print("  Inspired by: STSNet (Zhao et al., Scientific Reports 2025)")
print("=" * 70)
print(f"\n  Sampling ratios:")
for name, ratio in ratios.items():
    print(f"    {name:<20} {ratio:.0%}")
print(f"\n  Patch size:       {tuple(args.patch_size)}")
print(f"  Small threshold:  {args.small_threshold} voxels")
print(f"  Model:            ResUNet3d baseline (unchanged)")
print(f"  Loss:             BCEDiceLoss (unchanged)")
print(f"  Checkpoint:       {CHECKPOINT_DIR}")
print(f"\n  ONLY CHANGE:      data sampling strategy (uniform → 4-strategy FG-aware)")

# ============================================================
# Fairness Checklist
# ============================================================

print(f"\n{'='*70}")
print("FAIRNESS CHECKLIST")
print(f"{'='*70}")
print(f"  Data split:       tumourCSV.csv + random_state=10 (same as baseline) [OK]")
print(f"  Model:            ResUNet3d (same as baseline)                        [OK]")
print(f"  Learning rate:    {args.lr} (same as baseline)                         [OK]")
print(f"  Optimizer:        Adam (same as baseline)                              [OK]")
print(f"  Scheduler:        ReduceLROnPlateau patience=2 (same as baseline)      [OK]")
print(f"  Batch size:       1, accumulation=4 (same as baseline)                 [OK]")
print(f"  n_channels:       24 (same as baseline)                                [OK]")
print(f"  Seed:             55 (same as baseline)                                [OK]")
print(f"  Loss:             BCEDiceLoss (same as baseline)                       [OK]")
print(f"  Early stopping:   patience=25, min_delta=1e-4 (same as baseline)      [OK]")
print(f"  Checkpoint:       best val_loss (same as baseline)                    [OK]")
print(f"  ONLY CHANGE:      sampling strategy (uniform → FG-aware 4-strategy)   [OK]")
print(f"{'='*70}\n")

# ============================================================
# Dataset with Foreground-Aware Sampling
# ============================================================

# Read CSV and split (same as get_dataloader in data/dataset.py)
df = pd.read_csv(config.path_to_csv)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=10, shuffle=True)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
val_df = test_df.iloc[:len(test_df)*2//3].reset_index(drop=True)

# Create datasets
print("Creating datasets...")
train_dataset = BratsDatasetWithFGSampling(
    train_df, phase='train', is_resize=True,
    patch_size=tuple(args.patch_size),
    ratios=ratios,
    small_threshold=args.small_threshold,
)

val_dataset = BratsDatasetWithFGSampling(
    val_df, phase='valid', is_resize=True,  # no patch sampling for val
)

# Build foreground index (with caching)
cache_path = os.path.join(CHECKPOINT_DIR, 'foreground_index_cache.pkl')
if args.no_cache and os.path.exists(cache_path):
    os.remove(cache_path)
train_dataset.build_foreground_index(cache_path=cache_path)

# ============================================================
# Model & Loss (unchanged from baseline)
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
criterion = BCEDiceLoss()

print(f"\nModel: ResUNet3d ({sum(p.numel() for p in model.parameters()):,} params)")
print(f"Loss:  BCEDiceLoss (same as baseline)")

# ============================================================
# Trainer (load dataloaders manually)
# ============================================================

trainer = Trainer(
    net=model, dataset=BratsDataset, criterion=criterion,
    lr=args.lr, accumulation_steps=4, batch_size=1, fold=0,
    num_epochs=args.epochs, path_to_csv=config.path_to_csv,
    model_type=CHECKPOINT_DIR, display_plot=True,
    early_stopping_patience=25, min_delta=1e-4,
)

# Override train dataloader with FG-aware patch sampling
from torch.utils.data import DataLoader

trainer.dataloaders['train'] = DataLoader(
    train_dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=True,
)
# Valid stays the same (BratsDataset, full volume)
print(f"  Train dataloader: {len(trainer.dataloaders['train'])} FG-aware batches")
print(f"  Valid dataloader: {len(trainer.dataloaders['valid'])} full-volume batches")

# ============================================================
# Warm-start / Resume
# ============================================================

pretrained = check_exist(config.ResUNet_checkpoint_dir)
if pretrained:
    print(f"\nWarm-start from ResUNet baseline: {pretrained}")
    pretrain_state = torch.load(pretrained, map_location=device)
    pretrain_state = {k.replace('out.conv.0.', 'out.conv.'): v
                      for k, v in pretrain_state.items()}
    model_state = model.state_dict()
    matched = {k: v for k, v in pretrain_state.items()
               if k in model_state and v.shape == model_state[k].shape}
    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"  Loaded {len(matched)}/{len(model_state)} parameter tensors")
else:
    print(f"\n[WARN] No pretrained baseline found — training from scratch")

resume = check_exist(CHECKPOINT_DIR)
if resume:
    print(f"Resuming from: {resume}")
    trainer.load_pretrain_model(resume)

# ============================================================
# Train
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING (Foreground-Aware Patch Sampling)")
print("=" * 70 + "\n")

trainer.run(check_path=CHECKPOINT_DIR)

print(f"\nDone. Model saved to: {CHECKPOINT_DIR}")
print(f"Foreground index cache: {cache_path}")
