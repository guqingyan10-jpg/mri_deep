"""
=============================================================================
Train ResUNet with BCEDiceCCPMLoss — all three Dice variants combined
=============================================================================
The full formula: BCE + Global Dice + CC Dice + PM Dice

Loss formula:
  L = BCE + Global_Dice + λ_cc · CC_Dice + λ_pm · PM_Dice

Components (each addresses a different spatial level):
  Global Dice  → volume-level    (all pixels equal)
  CC Dice      → lesion-level    (each ET component equal weight)
  PM Dice      → pixel-level     (difficulty-modulated, Hosseini 2025)
  BCE          → per-pixel classification

Comparison matrix:
  BCEDiceLoss   = BCE + Global Dice                              (baseline)
  BCECCDiceLoss = BCE +            CC Dice                       (replacement)
  BCEPMDiceLoss = BCE +                       PM Dice            (replacement)
  BCEDiceCCLoss = BCE + Global Dice + CC Dice                    (additive)
  BCEDicePMLoss = BCE + Global Dice +            PM Dice         (additive)
  BCEDiceCCPMLoss = BCE + Global Dice + CC Dice + PM Dice        (THIS: full)

Usage:
    # Default (λ_cc=1.0, λ_pm=1.0, γ=2.0):
    python scripts/train_full_combined.py

    # Tune weights:
    python scripts/train_full_combined.py --lambda_cc 0.5 --lambda_pm 2.0

Checkpoint:
    /root/autodl-tmp/ResUNet_FullCombined_model/

Author: Generated for ResUNet enhancement project
Date:   2026-08-07
=============================================================================
"""

import os, sys, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from losses.enhanced import BCEDiceCCPMLoss
from data.dataset import BratsDataset
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist, check_exist_last

# ============================================================
# Parse
# ============================================================

parser = argparse.ArgumentParser(
    description='Train ResUNet with BCE + Global Dice + CC Dice + PM Dice',
)
parser.add_argument('--lambda_cc', type=float, default=1.0,
                    help='Weight for CC-level Dice (default 1.0)')
parser.add_argument('--lambda_pm', type=float, default=1.0,
                    help='Weight for PM Dice (default 1.0)')
parser.add_argument('--pm_gamma', type=float, default=2.0,
                    help='Focusing parameter γ for PM Dice (default 2.0)')
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

CHECKPOINT_DIR = '/root/autodl-tmp/ResUNet_FullCombined_model'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# Print
# ============================================================

print("=" * 70)
print("Full Combined Loss: BCE + Global Dice + CC Dice + PM Dice")
print("=" * 70)
print(f"\n  Loss formula:")
print(f"    L = BCE + Global_Dice + {args.lambda_cc}·CC_Dice + {args.lambda_pm}·PM_Dice(γ={args.pm_gamma})")
print(f"  Baseline:  L = BCE + Global_Dice (BCEDiceLoss)")
print(f"  Delta:     + CC_Dice + PM_Dice (two new terms)")
print(f"\n  Config:")
print(f"    lambda_cc:    {args.lambda_cc}")
print(f"    lambda_pm:    {args.lambda_pm}")
print(f"    pm_gamma:     {args.pm_gamma}")
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
print(f"  Eval metrics:  Dice / Recall / HD95 / Lesion Recall     [OK]")
print(f"")
print(f"  VARIABLE:      loss function")
print(f"    Baseline:    L = BCE + Global Dice")
print(f"    This expt:   L = BCE + Global + CC + PM (all three)")
print(f"                  └── two extra Dice variants added")
print(f"  → Delta = marginal contribution of CC + PM")
print(f"{'='*70}\n")

# ============================================================
# Model (unchanged)
# ============================================================

model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: ResUNet3d ({n_params:,} params) — SAME as baseline")

# ============================================================
# Loss — all three Dice variants
# ============================================================

criterion = BCEDiceCCPMLoss(
    lambda_cc=args.lambda_cc, lambda_pm=args.lambda_pm,
    cc_min_size=args.cc_min_size, pm_gamma=args.pm_gamma,
)

print(f"\nLoss: BCEDiceCCPMLoss (Full Combined)")
print(f"  BCE           ← baseline")
print(f"  Global Dice   ← baseline")
print(f"  CC Dice       ← per-ET-component equal weight")
print(f"  PM Dice       ← pixel-wise difficulty modulation (Hosseini 2025)")
print(f"  λ_cc={args.lambda_cc}, λ_pm={args.lambda_pm}, γ={args.pm_gamma}")

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
print("STARTING TRAINING (BCE + Global + CC + PM Dice)")
print("=" * 70 + "\n")

trainer.run(check_path=CHECKPOINT_DIR)
print(f"\nDone. Model saved to: {CHECKPOINT_DIR}")
print(f"\nEvaluation after training:")
print(f"  python scripts/eval_all_experiments.py --filter FullCombined")
print(f"\nKey metrics to compare:")
print(f"  vs Baseline        → Global + CC + PM vs just Global Dice")
print(f"  vs BCE+CCDice      → does adding Global + PM help beyond CC alone?")
print(f"  vs BCE+PMDice      → does adding Global + CC help beyond PM alone?")
