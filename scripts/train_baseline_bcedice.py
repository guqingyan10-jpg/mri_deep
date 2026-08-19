"""Train the ResUNet BCEDice baseline for one stability-test seed.

This script intentionally has the same optimizer, scheduler, data split,
batching, and early-stopping configuration as the existing experiments.  It
only resumes checkpoints stored in ``--checkpoint_dir`` so seeds cannot share
training state.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import BratsDataset
from losses.basics import BCEDiceLoss
from models.resunet3d import ResUNet3d
from training.config import check_exist_last, config, seed_everything
from training.trainer import Trainer


parser = argparse.ArgumentParser(
    description="Train one BCEDice ResUNet baseline for seed stability testing",
)
parser.add_argument("--seed", type=int, default=config.seed)
parser.add_argument("--checkpoint_dir", required=True,
                    help="Directory dedicated to this seed's baseline checkpoints")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

seed_everything(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = args.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 72)
print("Baseline ResUNet: BCEDiceLoss")
print("=" * 72)
print(f"  Seed:             {args.seed}")
print(f"  Checkpoint:       {CHECKPOINT_DIR}")
print(f"  Learning rate:    {args.lr}")
print("  Optimizer:        Adam")
print("  Scheduler:        ReduceLROnPlateau patience=2")
print("  Batch/accumulate: 1 / 4")
print("  Early stopping:   patience=25, min_delta=1e-4")
print("  Data split:       tumourCSV.csv + random_state=10")

model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
trainer = Trainer(
    net=model,
    dataset=BratsDataset,
    criterion=BCEDiceLoss(),
    lr=args.lr,
    accumulation_steps=4,
    batch_size=1,
    fold=0,
    num_epochs=args.epochs,
    path_to_csv=config.path_to_csv,
    model_type=CHECKPOINT_DIR,
    display_plot=True,
    early_stopping_patience=25,
    min_delta=1e-4,
)

resume = check_exist_last(CHECKPOINT_DIR)
if resume:
    print(f"Resuming this baseline from: {resume}")
    trainer.load_pretrain_model(resume)

print("\n" + "=" * 72)
print("STARTING BASELINE TRAINING")
print("=" * 72 + "\n")
trainer.run(check_path=CHECKPOINT_DIR)
print(f"\nDone. Baseline saved to: {CHECKPOINT_DIR}")
