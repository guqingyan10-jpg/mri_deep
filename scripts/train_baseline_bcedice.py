"""Train the ResUNet BCEDice baseline for one stability-test seed.

This script intentionally has the same optimizer, scheduler, data split,
batching, and early-stopping configuration as the existing experiments.  It
only resumes checkpoints stored in ``--checkpoint_dir`` so seeds cannot share
training state.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

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
completion_marker = os.path.join(CHECKPOINT_DIR, "training_complete.json")
# A prior marker must not unlock derived models if this run is interrupted.
if os.path.exists(completion_marker):
    os.remove(completion_marker)

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

best_checkpoints = [
    name for name in os.listdir(CHECKPOINT_DIR)
    if name.startswith("best_model_") and name.endswith(".pth")
]
if not best_checkpoints:
    raise RuntimeError(
        "Baseline training returned without producing best_model_*.pth; "
        "derived models will not be started."
    )
best_checkpoint = max(
    best_checkpoints,
    key=lambda name: int(name.rsplit("_", 1)[-1].split(".")[0]),
)
completion = {
    "status": "completed",
    "seed": args.seed,
    "epochs": args.epochs,
    "best_checkpoint": os.path.join(CHECKPOINT_DIR, best_checkpoint),
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
with open(completion_marker, "w", encoding="utf-8") as handle:
    json.dump(completion, handle, indent=2)
print(f"Baseline completion marker written: {CHECKPOINT_DIR}/training_complete.json")
print(f"\nDone. Baseline saved to: {CHECKPOINT_DIR}")
