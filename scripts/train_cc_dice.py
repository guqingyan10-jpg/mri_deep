"""
=============================================================================
Train with SLA-FB Step 2 Loss: Global Dice + CC-Level Dice + Weighted CE
=============================================================================
Reference: "Instance-level Dice Loss for Brain Tumor Segmentation"
  — Each ET connected component contributes equally to the loss.

Loss formula:
  L = L_Dice_global  +  lambda_cc · L_Dice_cc  +  gamma · L_CE

Comparisons:
  vs BCEDiceLoss (baseline):  global Dice no weights → CC-Dice + class weights
  vs DiceCEBoundary (V1):     Boundary loss replaced by CC-level Dice

Usage:
    # Default (CC-Dice + class weights):
    python scripts/train_cc_dice.py

    # Pure CC-Dice ablation (no class weights, isolate CC-Dice effect):
    python scripts/train_cc_dice.py --pure_cc --lambda_cc 1.0

    # Tune CC-Dice weight:
    python scripts/train_cc_dice.py --lambda_cc 0.5 --gamma_ce 0.5

Checkpoint:
    /root/autodl-tmp/ResUNet_CCDice_model/

Author: Generated for ResUNet enhancement project
Date:   2026-08-04
=============================================================================
"""

import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from losses.enhanced import DiceCCELoss
from losses.basics import BCEDiceLoss
from data.dataset import BratsDataset
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist

# ============================================================
# Parse
# ============================================================

parser = argparse.ArgumentParser(
    description='Train ResUNet with CC-Level Dice Loss (SLA-FB Step 2)',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--lambda_cc', type=float, default=1.0,
                    help='Weight for CC-level Dice term (default 1.0)')
parser.add_argument('--gamma_ce', type=float, default=0.5,
                    help='Weight for CE term (default 0.5)')
parser.add_argument('--class_weights', type=float, nargs=3,
                    default=[1.0, 3.0, 5.0],
                    help='[WT, TC, ET] class weights for CE')
parser.add_argument('--pure_cc', action='store_true',
                    help='Pure ablation: class_weights=[1,1,1], only CC-Dice changes')
parser.add_argument('--cc_min_size', type=int, default=10,
                    help='Min ET component voxels for CC-level Dice')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--from_scratch', action='store_true')

args = parser.parse_args()

# Pure CC-Dice ablation: remove class weight effect
if args.pure_cc:
    args.class_weights = [1.0, 1.0, 1.0]

# ============================================================
# Setup
# ============================================================

seed_everything(config.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Checkpoint name encodes the loss config
if args.pure_cc:
    CHECKPOINT_DIR = f'/root/autodl-tmp/ResUNet_CCDice_Pure_model'
else:
    CHECKPOINT_DIR = f'/root/autodl-tmp/ResUNet_CCDice_model'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Print info
# ============================================================

print("=" * 70)
print("SLA-FB Step 2: CC-Level Dice Loss")
print("  Reference: Instance-level Dice Loss for Brain Tumor Segmentation")
print("=" * 70)
print(f"\n  Loss formula:")
print(f"    L = L_Dice_global + {args.lambda_cc}·L_Dice_cc + {args.gamma_ce}·L_CE")
print(f"\n  Config:")
print(f"    lambda_cc:      {args.lambda_cc}")
print(f"    gamma_ce:       {args.gamma_ce}")
print(f"    class_weights:  {args.class_weights}  [WT, TC, ET]")
print(f"    cc_min_size:    {args.cc_min_size}")
print(f"    pure_cc (abln): {args.pure_cc}")
print(f"\n  Model:            ResUNet3d (unchanged)")
print(f"  ONLY CHANGE:      loss (BCEDiceLoss → DiceCCELoss)")
print(f"  Checkpoint:       {CHECKPOINT_DIR}")

# ============================================================
# Fairness Checklist
# ============================================================

ablation_label = 'PURE CC-Dice ablation (no class weights)' if args.pure_cc else f'CC-Dice + class weights [{args.class_weights}]'

print(f"\n{'='*70}")
print("FAIRNESS CHECKLIST")
print(f"{'='*70}")
print(f"  Data split:    tumourCSV.csv + random_state=10  [same as baseline]")
print(f"  Model:         ResUNet3d (same as baseline)    [same as baseline]")
print(f"  Learning rate: {args.lr}                        [same as baseline]")
print(f"  Optimizer:     Adam                            [same as baseline]")
print(f"  Scheduler:     ReduceLROnPlateau patience=2    [same as baseline]")
print(f"  Batch size:    1, accumulation=4               [same as baseline]")
print(f"  n_channels:    24                              [same as baseline]")
print(f"  Seed:          55                              [same as baseline]")
print(f"  ONLY CHANGE:   loss function")
print(f"    Baseline:  L = Dice + BCE  (no weights)")
print(f"    This expt: L = Dice + CC-Dice + Class-Weighted CE")
print(f"                 └── {ablation_label} ──┘")
print(f"{'='*70}\n")

# ============================================================
# Model (unchanged)
# ============================================================

model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
baseline_params = sum(p.numel() for p in model.parameters())
print(f"\nModel:       ResUNet3d ({baseline_params:,} params)")
print(f"  SAME architecture as baseline. Only the loss function is different.")

# ============================================================
# Loss — ONLY thing that changes
# ============================================================

criterion = DiceCCELoss(
    lambda_cc=args.lambda_cc,
    gamma_ce=args.gamma_ce,
    class_weights=args.class_weights,
    cc_min_size=args.cc_min_size,
)

print(f"\nLoss: DiceCCELoss")
print(f"  Global Dice:   same as baseline's DiceLoss")
print(f"  CC-Level Dice: per ET connected component, equal weight")
print(f"                 → small lesions get same vote as large ones")
print(f"  Weighted CE:   ET={args.class_weights[2]}×, TC={args.class_weights[1]}×, WT={args.class_weights[0]}×")
print(f"\n  Baseline loss: BCE + Dice (ALL pixels equal)")
print(f"  This loss:     Dice + CC-Dice + Class-Weighted CE (small lesions get extra signal)")

if args.pure_cc:
    print(f"\n  ★ PURE ABLATION: class_weights=[1,1,1]")
    print(f"    Only CC-level Dice separates this from BCEDiceLoss.")
    print(f"    Delta = net contribution of instance-level Dice alone.")

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
print("STARTING TRAINING (CC-Level Dice Loss)")
print("=" * 70 + "\n")

trainer.run(check_path=CHECKPOINT_DIR)
print(f"\nDone. Model saved to: {CHECKPOINT_DIR}")
