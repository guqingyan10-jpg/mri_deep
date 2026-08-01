"""
=============================================================================
Train ResUNet with Enhanced Loss: L = Dice + CE + lambda_b * Boundary
=============================================================================
Usage:
    python scripts/train_enhanced.py --lambda_b 0.1
    python scripts/train_enhanced.py --lambda_b 0.3
    python scripts/train_enhanced.py --lambda_b 0.5

Loss formula:
    L = alpha * DiceLoss + beta * CELoss + lambda_b * BoundaryLoss
    where alpha=1.0, beta=0.5 (fixed), lambda_b is tuned
    Class weights: WT=1.0, TC=3.0, ET=5.0

Checkpoints saved to:
    /root/autodl-tmp/ResUNet_Enhanced_lb{lambda_b}_model/

Author: Generated for ResUNet enhancement project
Date:   2026-08-01
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# If running from mri_deep subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from losses.enhanced import DiceCEBoundaryLoss
from data.dataset import BratsDataset
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist

# ============================================================
# Parse arguments
# ============================================================

parser = argparse.ArgumentParser(description='Train ResUNet with enhanced loss')
parser.add_argument('--lambda_b', type=float, required=True,
                    choices=[0.1, 0.3, 0.5],
                    help='Boundary loss weight (0.1, 0.3, or 0.5)')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate')
parser.add_argument('--from_scratch', action='store_true',
                    help='Train from scratch (ignore pretrained ResUNet checkpoint)')
args = parser.parse_args()

lambda_b = args.lambda_b

# ============================================================
# Configuration
# ============================================================

seed_everything(config.seed)

# Checkpoint directory per lambda_b value
CHECKPOINT_DIR = f'/root/autodl-tmp/ResUNet_Enhanced_lb{lambda_b}_model'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print(f"Training ResUNet with Enhanced Loss (lambda_b = {lambda_b})")
print("=" * 70)
print(f"\nLoss formula:")
print(f"  L = 1.0 * DiceLoss + 0.5 * CELoss + {lambda_b} * BoundaryLoss")
print(f"  Class weights: WT=1.0, TC=3.0, ET=5.0")
print(f"\nCheckpoint dir: {CHECKPOINT_DIR}")
print(f"Epochs: {args.epochs}")
print(f"Learning rate: {args.lr}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ============================================================
# Model
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
print(f"\nModel: ResUNet3d — {sum(p.numel() for p in model.parameters()):,} params")

# ============================================================
# Loss
# ============================================================

criterion = DiceCEBoundaryLoss(
    alpha=1.0,                      # Dice weight (fixed)
    beta=0.5,                       # CE weight (fixed)
    gamma=lambda_b,                 # Boundary weight ← THE TUNED PARAMETER
    class_weights=[1.0, 3.0, 5.0],  # WT=1.0, TC=3.0, ET=5.0
    bd_max_weight=5.0,              # BoundaryLoss: max weight at GT surface
    bd_alpha=1.0,                   # BoundaryLoss: distance decay per voxel
)

print(f"\nLoss: DiceCEBoundaryLoss")
print(f"  alpha={criterion.alpha}, beta={criterion.beta}, gamma={criterion.gamma}")
print(f"  class_weights: {criterion.ce.class_weights}")

# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    net=model,
    dataset=BratsDataset,
    criterion=criterion,
    lr=args.lr,
    accumulation_steps=4,
    batch_size=1,
    fold=0,
    num_epochs=args.epochs,
    path_to_csv=config.path_to_csv,
    model_type=CHECKPOINT_DIR,
    display_plot=True,
)

# ============================================================
# Load pretrained ResUNet baseline (optional warm-start)
# ============================================================

if not args.from_scratch:
    pretrained_path = check_exist(config.ResUNet_checkpoint_dir)
    if pretrained_path is not None:
        print(f"\nLoading pretrained ResUNet baseline from: {pretrained_path}")
        trainer.load_pretrain_model(pretrained_path)
    else:
        print(f"\n[WARN] No pretrained ResUNet found at {config.ResUNet_checkpoint_dir}")
        print("Training from scratch.")
else:
    print("\nTraining from scratch (--from_scratch)")

# Resume if already partially trained
resume_path = check_exist(CHECKPOINT_DIR)
if resume_path is not None:
    print(f"Resuming from checkpoint: {resume_path}")
    trainer.load_pretrain_model(resume_path)

# ============================================================
# Train
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70 + "\n")

trainer.run(check_path=CHECKPOINT_DIR)

print("\n" + "=" * 70)
print(f"Training complete! Model saved to: {CHECKPOINT_DIR}")
print("=" * 70)
