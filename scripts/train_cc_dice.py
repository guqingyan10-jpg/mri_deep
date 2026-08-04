"""
=============================================================================
Train with BCEDice + CC-Level Dice Loss (SLA-FB Step 2)
=============================================================================
Reference: "Instance-level Dice Loss for Brain Tumor Segmentation"

Loss formula:
  L = BCE + Dice_global + λ_cc · CCLevelDice

Single-variable change from baseline:
  BCEDiceLoss   = BCE + Global Dice
  BCEDiceCCLoss = BCE + Global Dice + λ_cc · CC-Level Dice
                                              └── only new term

Model, data, optimizer, lr, scheduler — all unchanged.

Evaluation tracks: Dice, Recall, HD95 curves per epoch.

Usage:
    # Default (λ_cc=1.0):
    python scripts/train_cc_dice.py

    # Tune CC-Dice weight:
    python scripts/train_cc_dice.py --lambda_cc 0.5

Checkpoint:
    /root/autodl-tmp/ResUNet_CCDice_model/

Author: Generated for ResUNet enhancement project
Date:   2026-08-04
=============================================================================
"""

import os, sys, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from losses.enhanced import BCEDiceCCLoss
from losses.basics import BCEDiceLoss
from data.dataset import BratsDataset
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist

# ============================================================
# Parse
# ============================================================

parser = argparse.ArgumentParser(
    description='Train ResUNet with BCEDice + CC-Level Dice Loss',
)
parser.add_argument('--lambda_cc', type=float, default=1.0,
                    help='Weight for CC-level Dice term (default 1.0)')
parser.add_argument('--cc_min_size', type=int, default=10,
                    help='Min ET component voxels for CC-level Dice (default 10)')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--from_scratch', action='store_true')

args = parser.parse_args()

# ============================================================
# Setup
# ============================================================

seed_everything(config.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_DIR = '/root/autodl-tmp/ResUNet_CCDice_model'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Print
# ============================================================

print("=" * 70)
print("SLA-FB Step 2: BCEDice + CC-Level Dice Loss")
print("  Reference: Instance-level Dice Loss for Brain Tumor Segmentation")
print("=" * 70)
print(f"\n  Loss formula:")
print(f"    L = BCE + Global_Dice + {args.lambda_cc} · CC_Level_Dice")
print(f"  Baseline:  L = BCE + Global_Dice (BCEDiceLoss)")
print(f"  ONLY new:  + {args.lambda_cc} · CC_Level_Dice")
print(f"\n  Config:")
print(f"    lambda_cc:    {args.lambda_cc}")
print(f"    cc_min_size:  {args.cc_min_size}")
print(f"    Model:        ResUNet3d (UNCHANGED)")
print(f"    Warm-start:   ResUNet baseline")
print(f"    Checkpoint:   {CHECKPOINT_DIR}")

# ============================================================
# Fairness Checklist
# ============================================================

print(f"\n{'='*70}")
print("FAIRNESS CHECKLIST — Single Variable Change")
print(f"{'='*70}")
print(f"  Data split:    tumourCSV.csv + random_state=10           [OK]")
print(f"  Model:         ResUNet3d                                 [OK]")
print(f"  Learning rate: {args.lr}                                 [OK]")
print(f"  Optimizer:     Adam                                      [OK]")
print(f"  Scheduler:     ReduceLROnPlateau patience=2             [OK]")
print(f"  Batch size:    1, accumulation=4                        [OK]")
print(f"  n_channels:    24                                        [OK]")
print(f"  Seed:          55                                        [OK]")
print(f"  Eval metrics:  Dice / Recall / HD95                     [OK]")
print(f"")
print(f"  VARIABLE:      loss function")
print(f"    Baseline:    L = BCE + Global Dice")
print(f"    This expt:   L = BCE + Global Dice + {args.lambda_cc}·CC_Dice")
print(f"                    └── PER-ET-COMPONENT Dice, equal weight")
print(f"  → Delta = net contribution of instance-level Dice")
print(f"{'='*70}\n")

# ============================================================
# Model (unchanged)
# ============================================================

model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: ResUNet3d ({n_params:,} params) — SAME as baseline")

# ============================================================
# Loss — ONLY thing that changes
# ============================================================

criterion = BCEDiceCCLoss(lambda_cc=args.lambda_cc, cc_min_size=args.cc_min_size)

print(f"\nLoss: BCEDiceCCLoss")
print(f"  BCE           ← same as baseline")
print(f"  Global Dice   ← same as baseline")
print(f"  CC-Level Dice ← NEW: per-connected-component, equal weight")
print(f"  λ_cc          = {args.lambda_cc}")

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
print("STARTING TRAINING (BCEDice + CC-Level Dice)")
print("=" * 70 + "\n")

trainer.run(check_path=CHECKPOINT_DIR)
print(f"\nDone. Model saved to: {CHECKPOINT_DIR}")
print(f"\nEvaluation metrics to check after training:")
print(f"  ET Dice — overall segmentation quality")
print(f"  ET Recall — FN reduction signal")
print(f"  ET HD95 — boundary quality")
print(f"  Lesion-wise Recall — per-lesion hit rate")
print(f"  Small-case ET Dice — small ET (bottom 25%) subset")
print(f"\n  Compare vs Baseline (BCEDiceLoss) to get Δ from CC-Level Dice")
