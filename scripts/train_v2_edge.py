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
from losses.enhanced import DiceCEBoundaryLoss
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
parser.add_argument('--lambda_b', type=float, default=0.3,
                    help='Boundary loss weight (default 0.3 from V1 best)')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--from_scratch', action='store_true')
args = parser.parse_args()

seed_everything(config.seed)

CHECKPOINT_DIR = f'/root/autodl-tmp/ResUNet_Edge_{args.fusion}_model'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
print(f"V2: ResUNet + Sobel Edge Branch (fusion={args.fusion})")
print("=" * 70)
print(f"\n  Loss:   1.0*Dice + 0.5*CE + {args.lambda_b}*Boundary (Kervadec 2019)")
print(f"  Fusion: {args.fusion}")
print(f"  Weights: WT=1, TC=3, ET=5")
print(f"  Edge:    Sobel 3D gradient magnitude → 4-level pyramid")
print(f"  Warm-start: ResUNet baseline")
print(f"  Checkpoint: {CHECKPOINT_DIR}")

# ============================================================
# Model
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResUNetEdge(in_channels=4, n_classes=3, n_channels=24,
                    fusion=args.fusion).to(device)
print(f"\nModel: ResUNetEdge (fusion={args.fusion}) — {sum(p.numel() for p in model.parameters()):,} params")

# ============================================================
# Loss (using best lambda_b from V1)
# ============================================================

criterion = DiceCEBoundaryLoss(
    alpha=1.0, beta=0.5, gamma=args.lambda_b,
    class_weights=[1.0, 3.0, 5.0],
    bd_max_weight=5.0, bd_alpha=1.0,
)

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
