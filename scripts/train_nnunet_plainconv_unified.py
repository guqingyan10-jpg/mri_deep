"""Train nnU-Net v2 PlainConvUNet under the repository's unified protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/root/autodl-tmp/nnUNet_v2_PlainConv_unified_model"),
    )
    return parser


def validate_registered_protocol(args: argparse.Namespace) -> None:
    """Prevent accidental drift from the already registered baseline recipe."""
    expected = {
        "epochs": 200,
        "lr": 5e-4,
        "batch_size": 1,
        "accumulation_steps": 4,
        "early_stopping_patience": 25,
        "min_delta": 1e-4,
    }
    changed = [
        f"{name}={getattr(args, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    if changed:
        raise SystemExit(
            "For the registered fair baseline, keep the unified protocol fixed: "
            + "; ".join(changed)
        )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    validate_registered_protocol(args)

    import torch

    from data.dataset import BratsDataset
    from losses.basics import BCEDiceLoss
    from models.nnunet_plainconv3d import NNUNetV2PlainConv3D
    from training.config import check_exist_last, config, seed_everything
    from training.trainer import Trainer

    seed_everything(args.seed)
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NNUNetV2PlainConv3D(in_channels=4, n_classes=3).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("=" * 76)
    print("nnU-Net v2 PlainConvUNet architecture — unified training protocol")
    print("=" * 76)
    print(f"Architecture backend: dynamic-network-architectures PlainConvUNet")
    print(f"Parameters:           {parameter_count:,}")
    print(f"Seed:                 {args.seed}")
    print(f"Split:                fixed 257 train / 74 valid / 37 test")
    print("Loss:                 BCE-Dice")
    print(f"Optimizer:            Adam, lr={args.lr}")
    print("Scheduler:            ReduceLROnPlateau(patience=2)")
    print("Batch/accumulation:   1 / 4")
    print("Selection:            lowest validation BCE-Dice loss")
    print(f"Epoch ceiling:        {args.epochs}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    trainer = Trainer(
        net=model,
        dataset=BratsDataset,
        criterion=BCEDiceLoss(),
        lr=args.lr,
        accumulation_steps=args.accumulation_steps,
        batch_size=args.batch_size,
        fold=0,
        num_epochs=args.epochs,
        path_to_csv=config.path_to_csv,
        model_type=str(checkpoint_dir),
        display_plot=True,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
    )

    resume_path = check_exist_last(str(checkpoint_dir))
    if resume_path:
        print(f"Resuming weights using the repository's legacy resume behavior: {resume_path}")
        trainer.load_pretrain_model(resume_path)
    trainer.run(check_path=str(checkpoint_dir))

    best = sorted(
        checkpoint_dir.glob("best_model_*.pth"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not best:
        raise RuntimeError("training ended without a best_model_*.pth checkpoint")
    print(f"Best validation checkpoint: {best[-1]}")


if __name__ == "__main__":
    main()
