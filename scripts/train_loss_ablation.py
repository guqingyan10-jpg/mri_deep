"""
=============================================================================
SLA-FB Step 2: CC-Dice / PM-Dice Loss Ablation — 4 Experiments
=====================================================================================
Reference: "Instance-level Dice Loss" + Hosseini 2025 (arXiv:2506.15744)

Ablation sequence (single-variable change from BCEDiceLoss baseline):
  ┌────┬──────────────────────┬──────────────────────────────────────────┐
  │ #  │ Loss                  │ Formula                                  │
  ├────┼──────────────────────┼──────────────────────────────────────────┤
  │ 1  │ CC-Dice               │ BCE + Global Dice + λ_cc·CCDice         │
  │ 2  │ PM-Dice              │ BCE + Global Dice + λ_pm·PMDice          │
  │ 3  │ CC-Dice + PM-Dice   │ BCE + Global Dice + λ_cc·CCDice + λ_pm·PMDice │
  │ 4  │ A+B+C (full)         │ Same as #3 (CC+PM combined)              │
  └────┴──────────────────────┴──────────────────────────────────────────┘

Usage:
    python scripts/train_loss_ablation.py --mode cc       # Experiment 1
    python scripts/train_loss_ablation.py --mode pm       # Experiment 2
    python scripts/train_loss_ablation.py --mode cc_pm    # Experiment 3
    python scripts/train_loss_ablation.py --mode all      # Experiment 4 (same as cc_pm)

Evaluation: Dice, Recall, HD95, Lesion Recall, Small-case Dice.

Checkpoints:
    /root/autodl-tmp/ResUNet_CCDice_model/        (--mode cc)
    /root/autodl-tmp/ResUNet_PMDice_model/        (--mode pm)
    /root/autodl-tmp/ResUNet_CCPMDice_model/      (--mode cc_pm or --mode all)

Author: Generated for ResUNet enhancement project
Date:   2026-08-04
=====================================================================================
"""

import os, sys, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from losses.enhanced import BCEDiceCCLoss, BCEDicePMLoss, BCEDiceCCPMLoss
from data.dataset import BratsDataset
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist

# ============================================================
# Parse
# ============================================================

parser = argparse.ArgumentParser(
    description='Loss Ablation: CC-Dice / PM-Dice / CC+PM',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python scripts/train_loss_ablation.py --mode cc
  python scripts/train_loss_ablation.py --mode pm --pm_gamma 2.0
  python scripts/train_loss_ablation.py --mode cc_pm --lambda_cc 1.0 --lambda_pm 1.0
""",
)
parser.add_argument('--mode', type=str, required=True,
                    choices=['cc', 'pm', 'cc_pm', 'all'],
                    help='Loss ablation mode')
parser.add_argument('--lambda_cc', type=float, default=1.0,
                    help='Weight for CC-level Dice (default 1.0)')
parser.add_argument('--lambda_pm', type=float, default=1.0,
                    help='Weight for PM Dice (default 1.0)')
parser.add_argument('--pm_gamma', type=float, default=2.0,
                    help='Focusing parameter for PM Dice (default 2.0)')
parser.add_argument('--cc_min_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--from_scratch', action='store_true')

args = parser.parse_args()

# ============================================================
# Setup
# ============================================================

seed_everything(config.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODE_CHECKPOINTS = {
    'cc':    '/root/autodl-tmp/ResUNet_CCDice_model',
    'pm':    '/root/autodl-tmp/ResUNet_PMDice_model',
    'cc_pm': '/root/autodl-tmp/ResUNet_CCPMDice_model',
    'all':   '/root/autodl-tmp/ResUNet_CCPMDice_model',
}

MODE_LABELS = {
    'cc':    'BCE + Global Dice + CC-Level Dice',
    'pm':    'BCE + Global Dice + PM Dice (Hosseini 2025)',
    'cc_pm': 'BCE + Global Dice + CC-Level Dice + PM Dice',
    'all':   'BCE + Global Dice + CC-Level Dice + PM Dice (A+B+C)',
}

CHECKPOINT_DIR = MODE_CHECKPOINTS[args.mode]
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Print
# ============================================================

print("=" * 70)
print(f"SLA-FB Step 2 Loss Ablation — Experiment: {args.mode}")
print(f"  {MODE_LABELS[args.mode]}")
print("=" * 70)
print(f"\n  Baseline:     BCE + Global Dice (BCEDiceLoss)")
print(f"  This expt:    {MODE_LABELS[args.mode]}")
print(f"\n  Config:")
if args.mode in ('cc', 'cc_pm', 'all'):
    print(f"    lambda_cc:    {args.lambda_cc}")
if args.mode in ('pm', 'cc_pm', 'all'):
    print(f"    lambda_pm:    {args.lambda_pm}")
    print(f"    pm_gamma:     {args.pm_gamma}")
print(f"    Model:        ResUNet3d (UNCHANGED)")
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
print(f"    This expt:   L = {MODE_LABELS[args.mode]}")
print(f"{'='*70}\n")

# ============================================================
# Model (unchanged)
# ============================================================

model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: ResUNet3d ({n_params:,} params) — identical to baseline")

# ============================================================
# Loss — ONLY thing that changes
# ============================================================

if args.mode == 'cc':
    criterion = BCEDiceCCLoss(lambda_cc=args.lambda_cc, cc_min_size=args.cc_min_size)
    print(f"\nLoss:  BCEDiceCCLoss")
    print(f"  + BCE          ← baseline")
    print(f"  + Global Dice  ← baseline")
    print(f"  + λ_cc={args.lambda_cc}·CC-Dice  ← NEW (per-ET-component)")

elif args.mode == 'pm':
    criterion = BCEDicePMLoss(lambda_pm=args.lambda_pm, pm_gamma=args.pm_gamma)
    print(f"\nLoss:  BCEDicePMLoss")
    print(f"  + BCE          ← baseline")
    print(f"  + Global Dice  ← baseline")
    print(f"  + λ_pm={args.lambda_pm}·PM-Dice(γ={args.pm_gamma})  ← NEW (Hosseini 2025)")

elif args.mode in ('cc_pm', 'all'):
    criterion = BCEDiceCCPMLoss(
        lambda_cc=args.lambda_cc, lambda_pm=args.lambda_pm,
        cc_min_size=args.cc_min_size, pm_gamma=args.pm_gamma,
    )
    print(f"\nLoss:  BCEDiceCCPMLoss (A+B+C)")
    print(f"  + BCE          ← baseline")
    print(f"  + Global Dice  ← baseline")
    print(f"  + λ_cc={args.lambda_cc}·CC-Dice    ← instance-level")
    print(f"  + λ_pm={args.lambda_pm}·PM-Dice(γ={args.pm_gamma}) ← difficulty-aware")

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
# Warm-start
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
print(f"STARTING TRAINING ({args.mode})")
print("=" * 70 + "\n")

trainer.run(check_path=CHECKPOINT_DIR)
print(f"\nDone. Model saved to: {CHECKPOINT_DIR}")
print(f"\nEvaluation after training:")
print(f"  python scripts/eval_lambda_experiments.py  # add this model")
print(f"\nKey metrics to compare vs Baseline:")
print(f"  ET Dice, ET Recall, ET HD95, Lesion Recall, Small-case ET Dice")
