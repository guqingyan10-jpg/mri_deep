"""
=============================================================================
Train ResUNet + FGFE (Frequency Guidance Feature Enhancement)
=============================================================================
Based on Yao et al., BraTS-UMamba, MICCAI 2025.

Laplacian pyramid on decoder features → cross-attention fusion.
Clean ablation: BCEDiceLoss, no class weights, same hyperparams as baseline.

Usage:
    python scripts/train_fgfe.py

Checkpoint:
    /root/autodl-tmp/ResUNet_FGFE_model/

Author: Generated for ResUNet enhancement project
Date:   2026-08-02
=============================================================================
"""

import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet_fgfe import ResUNetFGFE
from models.resunet3d import ResUNet3d
from losses.basics import BCEDiceLoss
from data.dataset import BratsDataset
from training.trainer import Trainer
from training.config import config, seed_everything, check_exist

CHECKPOINT_DIR = '/root/autodl-tmp/ResUNet_FGFE_model'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

seed_everything(config.seed)

print("=" * 70)
print("V2-FGFE: ResUNet + Frequency Guidance Feature Enhancement")
print("  Paper:  Yao et al., BraTS-UMamba, MICCAI 2025")
print("=" * 70)
print(f"  Loss:       BCEDiceLoss (same as baseline)")
print(f"  Weights:    no class weights (same as baseline)")
print(f"  Architecture: ResUNet encoder + FGFE decoder blocks")
print(f"  FGFE:        Laplacian pyramid → cross-attention → residual")
print(f"  ONLY CHANGE: ResUNet → ResUNetFGFE (+ FGFE at each decoder stage)")

# ============================================================
# Fairness checklist
# ============================================================
print(f"\n{'='*70}")
print("FAIRNESS CHECKLIST")
print(f"{'='*70}")
print(f"  Data split:    tumourCSV.csv + random_state=10  [same as baseline]")
print(f"  Learning rate: 5e-4                              [same as baseline]")
print(f"  Optimizer:     Adam                              [same as baseline]")
print(f"  Scheduler:     ReduceLROnPlateau patience=2      [same as baseline]")
print(f"  Batch size:    1, accumulation=4                 [same as baseline]")
print(f"  n_channels:    24                                [same as baseline]")
print(f"  Seed:          55                                [same as baseline]")
print(f"  Loss:          BCEDiceLoss                       [same as baseline]")
print(f"  Early stop:    patience=25, min_delta=1e-4       [same as baseline]")
print(f"  Checkpoint:    best val_loss                     [same as baseline]")
print(f"  ONLY CHANGE:   ResUNet → ResUNetFGFE")
print(f"{'='*70}\n")

# ============================================================
# Model
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResUNetFGFE(in_channels=4, n_classes=3, n_channels=24).to(device)

baseline_params = sum(p.numel() for p in ResUNet3d(4, 3, 24).parameters())
fgfe_params = sum(p.numel() for p in model.parameters())

print(f"Parameter count:")
print(f"  Baseline ResUNet:  {baseline_params:,}")
print(f"  ResUNetFGFE:       {fgfe_params:,}  (+{fgfe_params - baseline_params:,})")

# ============================================================
# Loss & Trainer
# ============================================================

criterion = BCEDiceLoss()
print(f"\nLoss: BCEDiceLoss (ORIGINAL — no modified class weights)")

trainer = Trainer(
    net=model, dataset=BratsDataset, criterion=criterion,
    lr=5e-4, accumulation_steps=4, batch_size=1, fold=0,
    num_epochs=200, path_to_csv=config.path_to_csv,
    model_type=CHECKPOINT_DIR, display_plot=True,
    early_stopping_patience=25, min_delta=1e-4,
)

# ============================================================
# Warm-start from ResUNet baseline
# ============================================================

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
    print(f"  Loaded {len(matched)}/{len(model_state)} encoder+decoder weights")
    print(f"  New FGFE layers: randomly initialized")
else:
    print(f"\n[WARN] No pretrained baseline found at {config.ResUNet_checkpoint_dir}")

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
