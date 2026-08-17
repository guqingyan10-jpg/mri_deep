"""
Train the final HF Concat Boundary combination model.

Architecture:
    Laplacian residual -> multi-scale EdgePyramid -> concat at dec1..dec4
    -> segmentation head + boundary auxiliary head

Loss:
    BCEDiceLoss(seg, GT) + boundary_weight * BCE(boundary, boundary_GT)

Fairness:
    The data split, seed, optimizer, learning rate, scheduler, batch size,
    gradient accumulation, early stopping, and checkpoint policy match the
    baseline and existing ablation scripts. A new run warm-starts from the
    baseline ResUNet best checkpoint; an interrupted run resumes from this
    experiment's latest checkpoint.

Usage:
    python scripts/train_hf_concat_boundary.py                      # w=0.3 (default)
    python scripts/train_hf_concat_boundary.py --boundary_weight 0.1
    python scripts/train_hf_concat_boundary.py --boundary_weight 0.05
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import BratsDataset
from losses.enhanced import BCEDiceWithBoundaryLoss
from models.resunet3d import ResUNet3d
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
from training.config import check_exist, check_exist_last, config, seed_everything
from training.trainer import Trainer


class HFConcatBoundaryTrainer(Trainer):
    """Pass both heads to the loss and only segmentation logits to metrics."""

    def _compute_loss_and_outputs(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        seg, boundary_pred = self.net(images)
        loss = self.criterion((seg, boundary_pred), targets)
        return loss, seg


parser = argparse.ArgumentParser(
    description="Train Laplacian multi-scale concat ResUNet with boundary supervision",
)
parser.add_argument("--boundary_weight", type=float, default=0.3,
                    help="Weight for boundary auxiliary BCE loss (default 0.3)")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--from_scratch", action="store_true")
args = parser.parse_args()

seed_everything(config.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.boundary_weight == 0.3:
    CHECKPOINT_DIR = "/root/autodl-tmp/ResUNet_HFConcatBoundary_model"
else:
    CHECKPOINT_DIR = f"/root/autodl-tmp/ResUNet_HFConcatBoundary_w{args.boundary_weight}_model"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("=" * 72)
print("Final Combination: Laplacian HF + Multi-scale Concat + Boundary Head")
print("=" * 72)
print("  Edge input:       Laplacian high-frequency residual I - blur(I)")
print("  Fusion:           EdgePyramid concat at dec1, dec2, dec3, dec4")
print(f"  Loss:             BCEDiceLoss + {args.boundary_weight} * boundary BCE")
print("  Warm-start:       baseline ResUNet best_model checkpoint")
print(f"  Checkpoint:       {CHECKPOINT_DIR}")

print("\n" + "=" * 72)
print("FAIRNESS CHECKLIST")
print("=" * 72)
print("  Data split:       tumourCSV.csv + random_state=10        [SAME]")
print(f"  Learning rate:    {args.lr}                               [SAME]")
print("  Optimizer:        Adam                                    [SAME]")
print("  Scheduler:        ReduceLROnPlateau patience=2           [SAME]")
print("  Batch/accumulate: 1 / 4                                   [SAME]")
print("  Base channels:    24                                      [SAME]")
print("  Seed:             55                                      [SAME]")
print("  Early stopping:   patience=25, min_delta=1e-4            [SAME]")
print("  Main loss:        BCEDiceLoss                             [SAME]")
print(f"  Added term:       {args.boundary_weight} * BCE(boundary_pred, boundary_GT)  [NEW]")
print("=" * 72 + "\n")

model = ResUNetHFConcatBoundary(
    in_channels=4,
    n_classes=3,
    n_channels=24,
).to(device)
criterion = BCEDiceWithBoundaryLoss(boundary_weight=args.boundary_weight)

baseline_params = sum(p.numel() for p in ResUNet3d(4, 3, 24).parameters())
model_params = sum(p.numel() for p in model.parameters())
print(f"Baseline parameters: {baseline_params:,}")
print(f"Combined parameters: {model_params:,} (+{model_params - baseline_params:,})")

trainer = HFConcatBoundaryTrainer(
    net=model,
    dataset=BratsDataset,
    criterion=criterion,
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

if not args.from_scratch:
    pretrained = check_exist(config.ResUNet_checkpoint_dir)
    if pretrained and not os.path.basename(pretrained).startswith("best_model_"):
        raise FileNotFoundError(
            "Fair warm-start requires baseline best_model_*.pth, but only found: "
            f"{pretrained}"
        )
    if pretrained:
        print(f"\nWarm-start from baseline best checkpoint: {pretrained}")
        pretrain_state = torch.load(pretrained, map_location=device)
        pretrain_state = {
            key.replace("out.conv.0.", "out.conv."): value
            for key, value in pretrain_state.items()
        }
        model_state = model.state_dict()
        matched = {
            key: value
            for key, value in pretrain_state.items()
            if key in model_state and value.shape == model_state[key].shape
        }
        model_state.update(matched)
        model.load_state_dict(model_state)
        print(f"  Loaded {len(matched)}/{len(model_state)} matching tensors")
        print("  Edge pyramid, expanded concat layers, and boundary head keep their initialization")
    else:
        raise FileNotFoundError(
            "Baseline best checkpoint is required for a fair warm-start: "
            f"{config.ResUNet_checkpoint_dir}. Restore best_model_*.pth or "
            "pass --from_scratch explicitly."
        )
else:
    print("\nTraining from scratch (--from_scratch)")

resume = check_exist_last(CHECKPOINT_DIR)
if resume:
    print(f"Resuming this experiment from: {resume}")
    trainer.load_pretrain_model(resume)

print("\n" + "=" * 72)
print("STARTING TRAINING")
print("=" * 72 + "\n")
trainer.run(check_path=CHECKPOINT_DIR)

print(f"\nDone. Model saved to: {CHECKPOINT_DIR}")
print("Evaluate with:")
print('  python scripts/eval_all_experiments.py --filter "HF Concat Boundary"')
