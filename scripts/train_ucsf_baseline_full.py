"""Train paired UCSF-BMSR ResUNet Baseline and AFBMS-ResUNet Full models.

The protocol mirrors the formal BraTS experiment: seed 55, Adam 5e-4,
BCEDice main loss, batch 1 with accumulation 4, ReduceLROnPlateau patience 2,
up to 200 epochs, and early stopping patience 25/min_delta 1e-4. Full is
warm-started from the paired external-dataset Baseline best checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ucsf_bmsr import UCSFPreparedDataset, get_ucsf_dataloader, read_manifest
from losses.basics import BCEDiceLoss
from losses.enhanced import BCEDiceWithBoundaryLoss
from models.resunet3d import ResUNet3d
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
from training.config import check_exist_last, seed_everything
from training.trainer import Trainer
from training.ucsf_trainer import UCSFBoundaryTrainer


def checkpoint_epoch(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def best_checkpoint(directory: Path) -> Path:
    checkpoints = list(directory.glob("best_model_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No best_model_*.pth in {directory}")
    return max(checkpoints, key=checkpoint_epoch)


def completion_path(directory: Path) -> Path:
    return directory / "training_complete.json"


def completed_checkpoint(directory: Path) -> Path | None:
    marker = completion_path(directory)
    if not marker.is_file():
        return None
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("status") != "completed":
        return None
    try:
        return best_checkpoint(directory)
    except FileNotFoundError:
        return None


def mark_complete(directory: Path, model_name: str, seed: int, epochs: int) -> Path:
    checkpoint = best_checkpoint(directory)
    metadata = {
        "status": "completed",
        "dataset": "UCSF-BMSR",
        "model": model_name,
        "seed": seed,
        "max_epochs": epochs,
        "best_checkpoint": str(checkpoint.resolve()),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    completion_path(directory).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return checkpoint


def load_matching_baseline(full_model, baseline_path: Path, device: str) -> tuple[int, int]:
    state = torch.load(baseline_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        state = state.state_dict()
    state = {
        key.replace("out.conv.0.", "out.conv."): value
        for key, value in state.items()
    }
    target = full_model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in target and value.shape == target[key].shape
    }
    target.update(matched)
    full_model.load_state_dict(target)
    return len(matched), len(target)


def make_trainer(
    *, model, criterion, manifest: Path, checkpoint_dir: Path,
    epochs: int, lr: float, boundary: bool,
):
    trainer_type = UCSFBoundaryTrainer if boundary else Trainer
    return trainer_type(
        net=model,
        dataset=UCSFPreparedDataset,
        criterion=criterion,
        lr=lr,
        accumulation_steps=4,
        batch_size=1,
        fold=0,
        num_epochs=epochs,
        path_to_csv=str(manifest),
        model_type=str(checkpoint_dir),
        display_plot=False,
        early_stopping_patience=25,
        min_delta=1e-4,
        dataloader_factory=get_ucsf_dataloader,
    )


def train_baseline(args, output: Path, device: str) -> Path:
    directory = output / f"seed{args.seed}" / "baseline"
    directory.mkdir(parents=True, exist_ok=True)
    existing = completed_checkpoint(directory)
    if existing:
        print(f"Baseline already completed: {existing}")
        return existing
    marker = completion_path(directory)
    if marker.exists():
        marker.unlink()
    seed_everything(args.seed)
    model = ResUNet3d(in_channels=4, n_classes=1, n_channels=24).to(device)
    trainer = make_trainer(
        model=model,
        criterion=BCEDiceLoss(),
        manifest=args.manifest,
        checkpoint_dir=directory,
        epochs=args.epochs,
        lr=args.lr,
        boundary=False,
    )
    resume = check_exist_last(str(directory))
    if resume:
        print(f"Resuming UCSF Baseline: {resume}")
        trainer.load_pretrain_model(resume)
    trainer.run(check_path=str(directory))
    checkpoint = mark_complete(directory, "baseline_resunet3d", args.seed, args.epochs)
    print(f"Baseline complete: {checkpoint}")
    return checkpoint


def train_full(args, output: Path, device: str, baseline: Path) -> Path:
    directory = output / f"seed{args.seed}" / "full"
    directory.mkdir(parents=True, exist_ok=True)
    existing = completed_checkpoint(directory)
    if existing:
        print(f"Full already completed: {existing}")
        return existing
    marker = completion_path(directory)
    if marker.exists():
        marker.unlink()
    seed_everything(args.seed)
    model = ResUNetHFConcatBoundary(
        in_channels=4,
        n_classes=1,
        n_channels=24,
        fusion="concat",
        multiscale_context_v2=True,
    ).to(device)
    trainer = make_trainer(
        model=model,
        criterion=BCEDiceWithBoundaryLoss(boundary_weight=0.1),
        manifest=args.manifest,
        checkpoint_dir=directory,
        epochs=args.epochs,
        lr=args.lr,
        boundary=True,
    )
    resume = check_exist_last(str(directory))
    if resume:
        print(f"Resuming UCSF Full: {resume}")
        trainer.load_pretrain_model(resume)
    else:
        matched, total = load_matching_baseline(model, baseline, device)
        print(f"Full warm-start: loaded {matched}/{total} shape-compatible tensors from {baseline}")
    trainer.run(check_path=str(directory))
    checkpoint = mark_complete(directory, "afbms_resunet", args.seed, args.epochs)
    alpha = model.multiscale_context.alpha.detach().cpu().reshape(-1).tolist()
    (directory / "multiscale_gates.txt").write_text(
        "\n".join(f"alpha={value:.10g}" for value in alpha) + "\n",
        encoding="utf-8",
    )
    print(f"Full complete: {checkpoint}")
    return checkpoint


def print_protocol(args, frame) -> None:
    scan_counts = Counter(frame["split"])
    patient_counts = frame.groupby("split")["patient_group"].nunique().to_dict()
    print("=" * 72)
    print("UCSF-BMSR paired external-dataset training")
    print("=" * 72)
    print(f"Manifest:          {args.manifest}")
    print(f"Scans:             {dict(scan_counts)}")
    print(f"Patients:          {patient_counts}")
    print("Cross-validation:  disabled; one fixed grouped 70/20/10 split")
    print("Inputs:            T1post, T1pre, FLAIR, subtraction")
    print("Target:            binary metastasis segmentation")
    print(f"Seed/lr/epochs:    {args.seed} / {args.lr} / {args.epochs}")
    print("Batch/accumulate:  1 / 4")
    print("Optimizer:         Adam; ReduceLROnPlateau patience=2")
    print("Early stopping:    patience=25, min_delta=1e-4")
    print("Baseline:          ResUNet3d, n_channels=24, BCE+Dice")
    print("Full:              LHFC+ABS(0.1)+AR-MSC V2, paired Baseline warm-start")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("/root/autodl-tmp/ucsf_bmsr_prepared/manifest.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/root/autodl-tmp/ucsf_baseline_full"),
    )
    parser.add_argument("--models", nargs="+", choices=("baseline", "full"),
                        default=("baseline", "full"))
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frame = read_manifest(args.manifest)
    print_protocol(args, frame)
    if args.dry_run:
        baseline = ResUNet3d(4, 1, 24)
        full = ResUNetHFConcatBoundary(
            in_channels=4, n_classes=1, n_channels=24,
            fusion="concat", multiscale_context_v2=True,
        )
        print(f"Baseline parameters: {sum(p.numel() for p in baseline.parameters()):,}")
        print(f"Full parameters:     {sum(p.numel() for p in full.parameters()):,}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_root.mkdir(parents=True, exist_ok=True)
    baseline_directory = args.output_root / f"seed{args.seed}" / "baseline"
    if "baseline" in args.models:
        baseline = train_baseline(args, args.output_root, device)
    else:
        baseline = completed_checkpoint(baseline_directory)
        if baseline is None:
            raise FileNotFoundError(
                "Full requires a completed paired Baseline in "
                f"{baseline_directory}"
            )
    if "full" in args.models:
        train_full(args, args.output_root, device, baseline)


if __name__ == "__main__":
    main()
