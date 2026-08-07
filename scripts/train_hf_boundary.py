"""
=============================================================================
Train ResUNet + HF Boundary Auxiliary Branch
=============================================================================
Strict single-variable change from Exp-0 baseline.

Model:  ResUNetHFBoundary (baseline encoder+decoder + HF boundary branch)
Loss:   BCEDiceWithBoundaryLoss = BCEDiceLoss(seg,GT) + 0.2*BCE(boundary,boundary_GT)
Data:   BratsDataset (standard random sampling, no class_weight)
Split:  tumourCSV.csv, seed=10

HF branch:
  - Fixed Sobel or Laplacian edge extraction on raw MRI (non-trainable)
  - 1x1x1 Conv3d for channel alignment
  - Add to decoder last layer, then dual seg_head + boundary_head
  - Loss: main = unchanged baseline BCEDiceLoss, aux = weighted boundary BCE

Reference:
    Yi et al., "Frequency-Aware Ensemble", BraTS 2025, arXiv:2509.19353

Usage:
    # Default (Laplacian edges, boundary_weight=0.3):
    python scripts/train_hf_boundary.py

    # Compare with w=0.2:
    python scripts/train_hf_boundary.py --boundary_weight 0.2

    # Laplacian edges:
    python scripts/train_hf_boundary.py --edge_type laplacian

    # Tune boundary weight:
    python scripts/train_hf_boundary.py --boundary_weight 0.1

Checkpoint:
    /root/autodl-tmp/ResUNet_HFBoundary_model/

Author: Generated for ResUNet enhancement project
Date:   2026-08-07
=============================================================================
"""

import os, sys, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet_hf_boundary import ResUNetHFBoundary
from losses.enhanced import BCEDiceWithBoundaryLoss
from data.dataset import BratsDataset
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist, check_exist_last


# ============================================================
# Custom Trainer: handles model returning (seg, boundary) tuple
# ============================================================

class HFBoundaryTrainer(Trainer):
    """
    Thin override: model returns (seg, boundary_pred) tuple.
    Loss receives full tuple; Meter logging receives seg only.
    """
    def _compute_loss_and_outputs(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        seg, boundary_pred = self.net(images)
        loss = self.criterion((seg, boundary_pred), targets)
        return loss, seg  # Meter only sees seg for Dice/IoU logging


# ============================================================
# Parse
# ============================================================

parser = argparse.ArgumentParser(
    description='Train ResUNet + HF Boundary auxiliary branch',
)
parser.add_argument('--edge_type', type=str, default='laplacian',
                    choices=['sobel', 'laplacian'],
                    help='Edge extraction method: sobel (1st deriv) or laplacian (2nd deriv)')
parser.add_argument('--boundary_weight', type=float, default=0.2,
                    help='Weight for boundary auxiliary BCE loss (default 0.2)')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--from_scratch', action='store_true')

args = parser.parse_args()

# ============================================================
# Setup
# ============================================================

seed_everything(config.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_DIR = f'/root/autodl-tmp/ResUNet_HFBoundary_w{args.boundary_weight}_model'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Print
# ============================================================

print("=" * 70)
print(f"ResUNet + HF Boundary Branch (edge_type={args.edge_type})")
print("=" * 70)
print(f"\n  Architecture:")
print(f"    Encoder+Decoder: 100% ResUNet3d (UNCHANGED)")
print(f"    HF Branch:       {args.edge_type} (FIXED, non-trainable)")
print(f"                     + 1x1x1 Conv channel align")
print(f"                     + ADD with decoder last output")
print(f"    Heads:           seg_head (original Out) + boundary_head (NEW)")
print(f"\n  Loss:")
print(f"    L = BCEDiceLoss(seg, GT) + {args.boundary_weight} * BCE(boundary, boundary_GT)")
print(f"    Main loss:       BCEDiceLoss (100% baseline)")
print(f"    Boundary weight: {args.boundary_weight}")
print(f"\n  Config:")
print(f"    edge_type:       {args.edge_type}")
print(f"    Model:           ResUNetHFBoundary ({5_804_000:,} params)")
print(f"    Warm-start:      ResUNet baseline (encoder+decoder only)")
print(f"    Checkpoint:      {CHECKPOINT_DIR}")

# ============================================================
# Fairness Checklist
# ============================================================

print(f"\n{'='*70}")
print("FAIRNESS CHECKLIST — Single Variable Change")
print(f"{'='*70}")
print(f"  Data split:    tumourCSV.csv + random_state=10           [OK]")
print(f"  Learning rate: {args.lr}                                 [OK]")
print(f"  Optimizer:     Adam                                      [OK]")
print(f"  Scheduler:     ReduceLROnPlateau patience=2             [OK]")
print(f"  Batch size:    1, accumulation=4                        [OK]")
print(f"  n_channels:    24                                        [OK]")
print(f"  Seed:          55                                        [OK]")
print(f"  Main loss:     BCEDiceLoss (IDENTICAL to baseline)      [OK]")
print(f"  Early stop:    patience=25, min_delta=1e-4              [OK]")
print(f"  Class weights: NONE (same as baseline)                  [OK]")
print(f"  Sampling:      standard random (same as baseline)       [OK]")
print(f"")
print(f"  VARIABLE:      HF boundary branch + auxiliary loss")
print(f"    Baseline:    ResUNet3d + BCEDiceLoss")
print(f"    This expt:   ResUNet3d + HF({args.edge_type}) boundary branch")
print(f"                 + BCEDiceLoss + {args.boundary_weight}*BCE(boundary)")
print(f"  → Only HF branch + boundary head are new. Everything else = baseline.")
print(f"{'='*70}\n")

# ============================================================
# Model
# ============================================================

model = ResUNetHFBoundary(
    in_channels=4, n_classes=3, n_channels=24,
    edge_type=args.edge_type,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: ResUNetHFBoundary ({n_params:,} params, {n_trainable:,} trainable)")
print(f"  Encoder+Decoder: ~5.76M (from baseline)")
print(f"  hf_align:        4→24, 1x1x1 = 96 params")
print(f"  boundary_head:   ~1.6K params")
print(f"  edge_extractor:  {sum(p.numel() for p in model.edge_extractor.parameters()):,} params (FROZEN)")

# ============================================================
# Loss
# ============================================================

criterion = BCEDiceWithBoundaryLoss(boundary_weight=args.boundary_weight)

print(f"\nLoss: BCEDiceWithBoundaryLoss")
print(f"  Main:      BCEDiceLoss (BCE + Global Dice)")
print(f"  Auxiliary: {args.boundary_weight} * BCE(boundary_pred, boundary_GT)")
print(f"  boundary_GT: extracted via GPU avg_pool3d erosion")

# ============================================================
# Trainer
# ============================================================

trainer = HFBoundaryTrainer(
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
        pretrain_state = torch.load(pretrained, map_location=device)

        # Key remapping: baseline 'out.*' → 'seg_head.*' (same architecture)
        pretrain_state = {k.replace('out.conv.', 'seg_head.conv.'): v
                          for k, v in pretrain_state.items()}

        model_state = model.state_dict()

        # Only load matching tensors (encoder+decoder+seg_head)
        # Skip hf_align, boundary_head, edge_extractor (random init)
        matched = {}
        skipped = []
        for k, v in pretrain_state.items():
            if k in model_state and v.shape == model_state[k].shape:
                matched[k] = v
            else:
                skipped.append(k)

        model_state.update(matched)
        model.load_state_dict(model_state)
        print(f"  Loaded {len(matched)}/{len(model_state)} parameter tensors")
        if skipped:
            new_keys = [k for k in model_state if any(
                p in k for p in ['hf_align', 'boundary_head', 'edge_extractor'])]
            print(f"  Skipped (random init): {len(skipped)} keys "
                  f"(hf_align, boundary_head, edge_extractor)")
    else:
        print(f"\n[WARN] No pretrained baseline found — training from scratch")
else:
    print("\nTraining from scratch (--from_scratch)")

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
print(f"\nEvaluation after training:")
print(f"  python scripts/eval_all_experiments.py --filter HFBoundary")
print(f"\nKey metrics (focus on HD95):")
print(f"  ET HD95  ← main metric: should drop if boundary branch works")
print(f"  TC HD95  ← secondary boundary metric")
print(f"  ET Dice, ET Recall ← reference (may also improve)")
print(f"  Lesion-wise Recall, Small-case ET Dice ← diagnostic")
