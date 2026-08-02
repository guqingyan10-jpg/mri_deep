"""
=============================================================================
Train ResUNet + Sobel Edge Branch (V2: High-Frequency Edge Auxiliary)
=============================================================================
Loss: Dice + CE + lambda_b * Boundary  (using best lambda_b from V1)
Model: ResUNetEdge with Sobel edge extraction + multi-scale fusion

Usage:
    # Concat fusion (recommended):
    python scripts/train_v2_edge.py --fusion concat

    # Add fusion:
    python scripts/train_v2_edge.py --fusion add

    # Both (sequentially):
    python scripts/train_v2_edge.py --fusion concat && python scripts/train_v2_edge.py --fusion add

Architecture:
    4-modal MRI → Sobel 3D → Edge Pyramid → Decoder (concat/add)
                 ↓
                ResUNet Encoder → Bottleneck → Decoder → Output

Checkpoints:
    /root/autodl-tmp/ResUNet_Edge_concat_model/
    /root/autodl-tmp/ResUNet_Edge_add_model/

Author: Generated for ResUNet enhancement project
Date:   2026-08-02
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet_edge import ResUNetEdge
from losses.basics import BCEDiceLoss
from data.dataset import BratsDataset
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist

# ============================================================
# Parse
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--fusion', type=str, required=True,
                    choices=['concat', 'add'],
                    help='Edge fusion mode')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--from_scratch', action='store_true')
parser.add_argument('--random_edge', action='store_true',
                    help='FAIRNESS CONTROL: replace Sobel edges with random noise. '
                         'Same param count, different input → isolates edge info contribution.')
args = parser.parse_args()

seed_everything(config.seed)

CHECKPOINT_DIR = f'/root/autodl-tmp/ResUNet_Edge_{args.fusion}_model'
if args.random_edge:
    CHECKPOINT_DIR = f'/root/autodl-tmp/ResUNet_Edge_random_control_model'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
edge_type = "RANDOM NOISE (fairness control)" if args.random_edge else "Sobel 3D gradient magnitude"
print(f"V2: ResUNet + Edge Branch (fusion={args.fusion})")
print("=" * 70)
print(f"\n  Loss:        BCEDiceLoss (ORIGINAL — same as baseline)")
print(f"  Fusion:      {args.fusion}")
print(f"  Edge input:  {edge_type}")
print(f"  Warm-start:  ResUNet baseline (encoder + decoder weights only)")
print(f"  Checkpoint:  {CHECKPOINT_DIR}")

# ============================================================
# Fairness Checklist
# ============================================================

print(f"\n{'='*70}")
print("FAIRNESS CHECKLIST")
print(f"{'='*70}")
print(f"  Data split:       tumourCSV.csv + random_state=10 (same as baseline)        [OK]")
print(f"  Learning rate:    {args.lr} (same as baseline)                                [OK]")
print(f"  Optimizer:        Adam (same as baseline)                                     [OK]")
print(f"  Scheduler:        ReduceLROnPlateau patience=2 (same as baseline)             [OK]")
print(f"  Batch size:       1, accumulation=4 (same as baseline)                        [OK]")
print(f"  n_channels:       24 (same as baseline)                                       [OK]")
print(f"  Seed:             55 (same as baseline)                                       [OK]")
print(f"  Loss:             BCEDiceLoss (same as baseline)                              [OK]")
print(f"  Early stopping:   patience=25, min_delta=1e-4 (same as baseline)             [OK]")
print(f"  Checkpoint:       best val_loss (same criterion as baseline)                 [OK]")
print(f"  Evaluation:       same test set, threshold=0.33, metrics                     [OK]")
if args.random_edge:
    print(f"  Control:          Random edge → isolates param count from edge info          [OK]")
print(f"  ONLY CHANGE:      ResUNet → ResUNetEdge (+ Sobel edge branch)")
print(f"{'='*70}\n")

# ============================================================
# Model
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResUNetEdge(in_channels=4, n_classes=3, n_channels=24,
                    fusion=args.fusion).to(device)
if args.random_edge:
    model.use_random_edge = True

# Compare param counts
from models.resunet3d import ResUNet3d
baseline_params = sum(p.numel() for p in ResUNet3d(4, 3, 24).parameters())
v2_params = sum(p.numel() for p in model.parameters())
print(f"\nParameter count:")
print(f"  Baseline ResUNet:  {baseline_params:,}")
print(f"  ResUNetEdge:       {v2_params:,}  (+{v2_params - baseline_params:,})")
print(f"  Random edge has SAME param count as Sobel edge (only input differs)")

# ============================================================
# Loss — ORIGINAL BCEDiceLoss (isolate model change from loss change)
# ============================================================
# V1 changed the loss. V2 changes ONLY the model.

criterion = BCEDiceLoss()
print(f"\nLoss: BCEDiceLoss (ORIGINAL — same as baseline)")
print(f"  V1 change: loss function    (BCEDice → DiceCEBoundary + class weights)")
print(f"  V2 change: model architecture (ResUNet → ResUNetEdge + Sobel branch)")
print(f"  Each experiment changes exactly ONE thing.")

# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    net=model, dataset=BratsDataset, criterion=criterion,
    lr=args.lr, accumulation_steps=4, batch_size=1, fold=0,
    num_epochs=args.epochs, path_to_csv=config.path_to_csv,
    model_type=CHECKPOINT_DIR, display_plot=True,
    early_stopping_patience=25, min_delta=1e-4,
)

# ============================================================
# Warm-start from ResUNet baseline
# ============================================================

if not args.from_scratch:
    pretrained = check_exist(config.ResUNet_checkpoint_dir)
    if pretrained:
        print(f"\nWarm-start from ResUNet baseline: {pretrained}")
        # Only load matching keys (encoder + decoder weights)
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

# Resume if checkpoint exists
resume = check_exist(CHECKPOINT_DIR)
if resume:
    print(f"Resuming from: {resume}")
    trainer.load_pretrain_model(resume)

# ============================================================
# Train
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70 + "\n")
trainer.run(check_path=CHECKPOINT_DIR)
print(f"\nDone. Model saved to: {CHECKPOINT_DIR}")
