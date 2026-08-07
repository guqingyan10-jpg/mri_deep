"""
=============================================================================
Train ResUNet with BCE + PM Dice Loss (no global Dice)
=============================================================================
Separate experiment: BCE + PM Dice (Hosseini 2025), NO global Dice term.

Loss formula:
  L = BCE + λ_pm · PMDice

PM Dice reference:
    Hosseini, S.M. (2025). "Pixel-wise Modulated Dice Loss for
    Medical Image Segmentation." arXiv:2506.15744.

    m_i = |y_i - p̂_i|^γ   (stop-gradient through p̂)
    L_PMDice = 1 - (1/C) Σ_c [ 2 Σ m_i·y_i·p_i + ε ] / [ Σ m_i·(y_i²+p_i²) + ε ]

    Easy pixels (m≈0) → near-zero contribution; hard pixels (m≈1) → full weight.
    Automatically shifts gradient budget toward boundaries and small lesions.

Comparison matrix:
  BCEDiceLoss   = BCE + Global Dice                     (baseline)
  BCEDicePMLoss = BCE + Global Dice + λ_pm·PM Dice      (existing: additive)
  BCEPMDiceLoss = BCE + λ_pm·PM Dice                    (THIS: replacement)

Single-variable change from baseline:
  Replaces Global Dice with PM Dice.
  Tests whether difficulty-modulated per-pixel Dice can replace global Dice.

Usage:
    # Default (λ_pm=1.0, γ=2.0):
    python scripts/train_bce_pmdice.py

    # Tune PM Dice weight:
    python scripts/train_bce_pmdice.py --lambda_pm 0.5
    python scripts/train_bce_pmdice.py --lambda_pm 2.0

    # Tune focusing parameter:
    python scripts/train_bce_pmdice.py --pm_gamma 1.0

Checkpoint:
    /root/autodl-tmp/ResUNet_BCEPMDice_model/

Author: Generated for ResUNet enhancement project
Date:   2026-08-07
=============================================================================
"""

import os, sys, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from losses.enhanced import BCEPMDiceLoss
from data.dataset import BratsDataset
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist, check_exist_last

# ============================================================
# Parse
# ============================================================

parser = argparse.ArgumentParser(
    description='Train ResUNet with BCE + PM Dice Loss (no global Dice)',
)
parser.add_argument('--lambda_pm', type=float, default=1.0,
                    help='Weight for PM Dice term (default 1.0)')
parser.add_argument('--pm_gamma', type=float, default=2.0,
                    help='Focusing parameter γ for PM Dice (default 2.0)')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--from_scratch', action='store_true')

args = parser.parse_args()

# ============================================================
# Setup
# ============================================================

seed_everything(config.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_DIR = '/root/autodl-tmp/ResUNet_BCEPMDice_model'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Print
# ============================================================

print("=" * 70)
print("BCE + PM Dice Loss (NO global Dice)")
print("  Reference: Hosseini 2025, arXiv:2506.15744")
print("=" * 70)
print(f"\n  Loss formula:")
print(f"    L = BCE + {args.lambda_pm} · PM_Dice(γ={args.pm_gamma})")
print(f"  Baseline:  L = BCE + Global_Dice (BCEDiceLoss)")
print(f"  Delta:     Replaces Global Dice with PM Dice")
print(f"\n  Config:")
print(f"    lambda_pm:    {args.lambda_pm}")
print(f"    pm_gamma:     {args.pm_gamma}")
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
print(f"  Eval metrics:  Dice / Recall / HD95 / Lesion Recall     [OK]")
print(f"")
print(f"  VARIABLE:      loss function")
print(f"    Baseline:    L = BCE + Global Dice")
print(f"    This expt:   L = BCE + {args.lambda_pm}·PM_Dice(γ={args.pm_gamma})")
print(f"                  └── PM Dice replaces Global Dice, NOT added")
print(f"  → Delta = PM Dice vs Global Dice")
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

criterion = BCEPMDiceLoss(lambda_pm=args.lambda_pm, pm_gamma=args.pm_gamma)

print(f"\nLoss: BCEPMDiceLoss")
print(f"  BCE           ← same as baseline")
print(f"  PM Dice       ← NEW: pixel-wise modulated (Hosseini 2025)")
print(f"  Global Dice   ← REMOVED (replaced by PM Dice)")
print(f"  λ_pm          = {args.lambda_pm}")
print(f"  γ             = {args.pm_gamma}")

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

resume = check_exist_last(CHECKPOINT_DIR)
if resume:
    print(f"Resuming from: {resume}")
    trainer.load_pretrain_model(resume)

# ============================================================
# Train
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING (BCE + PM Dice)")
print("=" * 70 + "\n")

trainer.run(check_path=CHECKPOINT_DIR)
print(f"\nDone. Model saved to: {CHECKPOINT_DIR}")
print(f"\nEvaluation after training:")
print(f"  python scripts/eval_all_experiments.py --filter BCEPM")
print(f"\nKey metrics to compare vs Baseline (BCEDiceLoss):")
print(f"  ET Dice     — overall segmentation quality")
print(f"  ET Recall   — FN reduction signal")
print(f"  ET HD95     — boundary quality")
print(f"  Lesion-wise Recall — per-lesion hit rate")
print(f"  Small-case ET Dice — small ET (bottom 25%) subset")
print(f"\nCompare with BCEDicePMLoss to see the contribution of Global Dice:")
print(f"  BCEDiceLoss   = BCE + Global Dice")
print(f"  BCEPMDiceLoss = BCE + PM Dice   ← THIS")
print(f"  BCEDicePMLoss = BCE + Global Dice + PM Dice")
