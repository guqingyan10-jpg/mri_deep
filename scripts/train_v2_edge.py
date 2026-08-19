"""
=============================================================================
Train ResUNet + Sobel Edge Branch (V2: High-Frequency Edge Auxiliary)
=============================================================================
Loss: Dice + CE + lambda_b * Boundary  (using best lambda_b from V1)
Model: ResUNetEdge with Sobel edge extraction + multi-scale fusion

Usage:
    # Sobel edge (1st derivative):
    python scripts/train_v2_edge.py --fusion concat --edge_type sobel

    # Laplacian edge (2nd derivative):
    python scripts/train_v2_edge.py --fusion concat --edge_type laplacian

    # Random control (fairness ablation):
    python scripts/train_v2_edge.py --fusion concat --edge_type random

    # Add fusion:
    python scripts/train_v2_edge.py --fusion add --edge_type sobel

Architecture:
    4-modal MRI → Sobel 3D → Edge Pyramid → Decoder (concat/add)
                 ↓
                ResUNet Encoder → Bottleneck → Decoder → Output

Checkpoints:
    /root/autodl-tmp/ResUNet_Edge_concat_sobel_model/
    /root/autodl-tmp/ResUNet_Edge_concat_laplacian_model/
    /root/autodl-tmp/ResUNet_Edge_concat_random_model/
    /root/autodl-tmp/ResUNet_Edge_add_sobel_model/
    ...

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
from training.config import config, seed_everything, check_exist, check_exist_last

# ============================================================
# Parse
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('--fusion', type=str, required=True,
                    choices=['concat', 'add'],
                    help='Edge fusion mode')
parser.add_argument('--edge_type', type=str, default='sobel',
                    choices=['sobel', 'laplacian', 'random'],
                    help='Edge extraction method. '
                         'sobel: 1st-derivative gradient magnitude; '
                         'laplacian: 2nd-derivative high-freq residual (I-blur(I)); '
                         'random: fairness control (noise instead of edges)')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=config.seed,
                    help='Random seed for this independent run')
parser.add_argument('--checkpoint_dir', type=str, default=None,
                    help='Directory dedicated to this experiment run')
parser.add_argument('--baseline_checkpoint', type=str, default=None,
                    help='Exact baseline best_model checkpoint for warm-start')
parser.add_argument('--from_scratch', action='store_true')
args = parser.parse_args()

seed_everything(args.seed)

DEFAULT_CHECKPOINT_DIR = f'/root/autodl-tmp/ResUNet_Edge_{args.fusion}_{args.edge_type}_model'
CHECKPOINT_DIR = args.checkpoint_dir or DEFAULT_CHECKPOINT_DIR
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 70)
edge_labels = {
    'sobel':     'Sobel 3D gradient magnitude (1st derivative)',
    'laplacian': 'Laplacian high-freq residual I-blur(I) (2nd derivative)',
    'random':    'RANDOM NOISE (fairness control — isolates param count from edge info)',
}
print(f"V2: ResUNet + Edge Branch (fusion={args.fusion}, edge={args.edge_type})")
print("=" * 70)
print(f"\n  Loss:        BCEDiceLoss (ORIGINAL — same as baseline)")
print(f"  Fusion:      {args.fusion}")
print(f"  Edge input:  {edge_labels[args.edge_type]}")
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
print(f"  Seed:             {args.seed}                                                   [OK]")
print(f"  Loss:             BCEDiceLoss (same as baseline)                              [OK]")
print(f"  Early stopping:   patience=25, min_delta=1e-4 (same as baseline)             [OK]")
print(f"  Checkpoint:       best val_loss (same criterion as baseline)                 [OK]")
print(f"  Evaluation:       same test set, threshold=0.33, metrics                     [OK]")
if args.edge_type == 'random':
    print(f"  Control:          Random edge → isolates param count from edge info          [OK]")
elif args.edge_type == 'laplacian':
    print(f"  Control:          Laplacian (2nd deriv) vs Sobel (1st deriv) → same param   [OK]")
print(f"  ONLY CHANGE:      ResUNet → ResUNetEdge (+ {args.edge_type} edge branch)")
print(f"{'='*70}\n")

# ============================================================
# Model
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResUNetEdge(in_channels=4, n_classes=3, n_channels=24,
                    fusion=args.fusion, edge_type=args.edge_type).to(device)

# Compare param counts
from models.resunet3d import ResUNet3d
baseline_params = sum(p.numel() for p in ResUNet3d(4, 3, 24).parameters())
v2_params = sum(p.numel() for p in model.parameters())
print(f"\nParameter count:")
print(f"  Baseline ResUNet:  {baseline_params:,}")
print(f"  ResUNetEdge:       {v2_params:,}  (+{v2_params - baseline_params:,})")
if args.edge_type == 'random':
    print(f"  [CONTROL] Same param count as Sobel/Laplacian — only input differs (noise)")
elif args.edge_type == 'laplacian':
    print(f"  [ABLATION] Same param count as Sobel — only derivative order differs (1st vs 2nd)")

# ============================================================
# Loss — ORIGINAL BCEDiceLoss (isolate model change from loss change)
# ============================================================
# V1 changed the loss. V2 changes ONLY the model.

criterion = BCEDiceLoss()
print(f"\nLoss: BCEDiceLoss (ORIGINAL — same as baseline)")
print(f"  V1 change: loss function    (BCEDice → DiceCEBoundary + class weights)")
print(f"  V2 change: model architecture (ResUNet → ResUNetEdge + {args.edge_type} edge branch)")
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
    pretrained = args.baseline_checkpoint or check_exist(config.ResUNet_checkpoint_dir)
    if args.baseline_checkpoint and not os.path.isfile(args.baseline_checkpoint):
        raise FileNotFoundError(
            f"Explicit baseline checkpoint does not exist: {args.baseline_checkpoint}"
        )
    if args.baseline_checkpoint and not os.path.basename(pretrained).startswith('best_model_'):
        raise ValueError(
            "Fair warm-start requires a baseline best_model_*.pth checkpoint, got: "
            f"{pretrained}"
        )
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
resume = check_exist_last(CHECKPOINT_DIR)
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
