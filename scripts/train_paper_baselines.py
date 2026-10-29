"""Train pinned DoubleBlock-ViT or SuperLightNet on the FULL model's BraTS split.

Run from the repository root.  See docs/PAPER_BASELINES.md before using --crop-size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import BratsDataset, get_dataloader
from losses.basics import BCEDiceLoss
from models.paper_baselines import PaperBaseline, SOURCES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=tuple(SOURCES), required=True)
    p.add_argument("--csv", type=Path, default=ROOT / "tumourCSV.csv")
    p.add_argument("--source-dir", type=Path, default=Path("/root/autodl-tmp/paper_baseline_sources"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=55)
    p.add_argument("--crop-size", type=int, nargs=3, metavar=("D", "H", "W"),
                   help="Optional memory-saving train crop; changes FULL's full-volume protocol")
    p.add_argument("--preflight", action="store_true", help="Check split, one real training case, upstream model and forward pass")
    p.add_argument("--resume", action="store_true", help="Resume only this model's interrupted run from last_state.pth")
    p.add_argument("--device", default=None)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_manifest(loaders):
    result = {phase: loader.dataset.df["Brats20ID"].astype(str).tolist()
              for phase, loader in loaders.items()}
    if not all(result.values()):
        raise ValueError("empty BraTS split")
    if len(set(sum(result.values(), []))) != sum(map(len, result.values())):
        raise ValueError("BraTS splits overlap or contain duplicate IDs")
    return result


def check_case_paths(loaders):
    for phase, loader in loaders.items():
        for row in loader.dataset.df.itertuples(index=False):
            folder, case_id = Path(row.path), str(row.Brats20ID)
            for suffix in ("_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii", "_seg.nii"):
                path = folder / (case_id + suffix)
                if not path.is_file():
                    raise FileNotFoundError(f"{phase}: {path}")


def crop_pair(images, targets, size):
    # BratsDataset already applies the FULL model's fixed [40:210,40:210,20:120]
    # volume crop, modality normalization and [WT,TC,ET] mask conversion.
    shape = images.shape[2:]
    if any(s > n for s, n in zip(size, shape)):
        raise ValueError(f"crop {size} exceeds project volume {shape}")
    starts = [random.randint(0, n-s) for n, s in zip(shape, size)]
    slices = tuple(slice(a, a+s) for a, s in zip(starts, size))
    return images[(..., *slices)], targets[(..., *slices)]


def forward_for_validation(model, images, crop_size):
    if crop_size is None:
        return model(images)
    return sliding_window_inference(
        images, roi_size=tuple(crop_size), sw_batch_size=1,
        predictor=model, overlap=0.25, mode="constant",
    )


def atomic_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError("epochs and learning rate must be positive")
    if args.crop_size is not None and any(d <= 0 or d % 32 for d in args.crop_size):
        raise ValueError("--crop-size dimensions must be positive multiples of 32")
    if args.resume and args.preflight:
        raise ValueError("choose --resume or --preflight")
    args.csv = args.csv.expanduser().resolve()
    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)
    output_dir = (args.output_dir or Path("/root/autodl-tmp/paper_baselines") / args.model / f"seed_{args.seed}")
    output_dir = output_dir.expanduser().resolve()
    if args.device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; refusing to start full-volume training on CPU. "
                "Check nvidia-smi and the installed PyTorch CUDA build. "
                "For a CPU-only architecture check, pass --preflight --device cpu."
            )
        device = torch.device("cuda")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; check the NVIDIA driver and PyTorch CUDA build")
        if device.type != "cuda" and not args.preflight:
            raise RuntimeError("full-volume training requires CUDA; CPU is supported only for --preflight")
    seed_all(args.seed)

    loaders = {phase: get_dataloader(BratsDataset, str(args.csv), phase,
                                     batch_size=1, num_workers=0)
               for phase in ("train", "valid", "test")}
    manifest = split_manifest(loaders)
    check_case_paths(loaders)
    if args.preflight or args.crop_size is not None:
        sample = loaders["train"].dataset[0]
        image_shape = tuple(sample["image"].shape)
        mask_shape = tuple(sample["mask"].shape)
        if image_shape != (4, 100, 170, 170) or mask_shape != (3, 100, 170, 170):
            raise RuntimeError(f"unexpected FULL dataset tensor shape: {image_shape}, {mask_shape}")
        if not np.isfinite(sample["image"]).all() or not np.isfinite(sample["mask"]).all():
            raise RuntimeError("non-finite values in first training case")
        if args.crop_size and any(s > n for s, n in zip(args.crop_size, image_shape[1:])):
            raise ValueError(f"crop {args.crop_size} exceeds FULL tensor shape {image_shape[1:]}")
        print(f"BraTS sample: image={image_shape}, mask={mask_shape}")
    model = PaperBaseline(args.model, args.source_dir.expanduser()).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.model == "doubleblock" and not 5_000_000 <= parameters <= 10_000_000:
        raise RuntimeError(f"unexpected DoubleBlock parameter count: {parameters}")
    if args.model == "superlight" and not 2_000_000 <= parameters <= 4_000_000:
        raise RuntimeError(f"unexpected SuperLight parameter count: {parameters}")

    protocol = {
        "model": args.model, "upstream_url": SOURCES[args.model][0],
        "upstream_commit": SOURCES[args.model][1], "csv": str(args.csv),
        "csv_sha256": sha256(args.csv), "split": manifest,
        "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
        "initialization": "from scratch, author model initialization; no ResUNet or paper checkpoint",
        "resume_policy": "only last_state.pth from the identical model run, never another model",
        "optimizer": "Adam(default betas, no weight decay)",
        "scheduler": "ReduceLROnPlateau(mode=min, patience=2)",
        "loss": "project BCEDiceLoss on [WT,TC,ET]",
        "batch_size": 1, "gradient_accumulation": 4,
        "incomplete_accumulation_group": "carried to next epoch, matching training.trainer.Trainer",
        "early_stopping_patience": 25, "min_delta": 1e-4,
        "preprocess": "data.dataset.BratsDataset (fixed crop, min-max, no augmentation)",
        "crop_size": args.crop_size, "parameters": parameters,
        "region_order": ["WT", "TC", "ET"],
    }
    print(json.dumps({k: v for k, v in protocol.items() if k != "split"}, indent=2))
    print("split sizes:", {k: len(v) for k, v in manifest.items()})
    if args.preflight:
        with torch.no_grad():
            probe = torch.zeros((1, 4, 32, 32, 32), device=device)
            result = model(probe)
            if result.shape != (1, 3, 32, 32, 32) or not torch.isfinite(result).all():
                raise RuntimeError(f"upstream forward failed: {tuple(result.shape)}")
        print("Preflight passed. No training or validation/test inference was run.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old != protocol:
            raise RuntimeError(f"run protocol changed; use a new --output-dir: {protocol_path}")
    elif args.resume:
        raise FileNotFoundError(protocol_path)
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    first_epoch, best_loss, bad_epochs = 1, float("inf"), 0
    last_path = output_dir / "last_state.pth"
    if args.resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_epoch = state["epoch"] + 1
        best_loss, bad_epochs = state["best_loss"], state["bad_epochs"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for name, parameter in model.named_parameters():
            gradient = state["pending_gradients"].get(name)
            parameter.grad = gradient.to(device) if gradient is not None else None
        print(f"Resuming from epoch {first_epoch}")
    elif last_path.exists():
        raise RuntimeError(f"existing run at {output_dir}; pass --resume or choose a new output dir")

    log_path = output_dir / "train_log.csv"
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        train_total = 0.0
        for i, batch in enumerate(loaders["train"]):
            images = batch["image"].float().to(device)
            targets = batch["mask"].float().to(device)
            if args.crop_size:
                images, targets = crop_pair(images, targets, args.crop_size)
            logits = model(images).contiguous()
            loss = criterion(logits, targets.contiguous())
            (loss / 4).backward()
            if (i + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_total += loss.item()
            if (i + 1) % 20 == 0:
                print(f"epoch {epoch} train {i+1}/{len(loaders['train'])} loss={train_total/(i+1):.5f}", flush=True)
        # Match the FULL Trainer: any incomplete group remains pending until
        # the first complete accumulation group in the following epoch.

        model.eval()
        valid_total = 0.0
        with torch.no_grad():
            for batch in loaders["valid"]:
                images = batch["image"].float().to(device)
                targets = batch["mask"].float().to(device)
                logits = forward_for_validation(model, images, args.crop_size).contiguous()
                valid_total += criterion(logits, targets.contiguous()).item()
        train_loss = train_total / len(loaders["train"])
        valid_loss = valid_total / len(loaders["valid"])
        scheduler.step(valid_loss)
        improved = valid_loss < best_loss - 1e-4
        if improved:
            best_loss, bad_epochs = valid_loss, 0
            for old in output_dir.glob("best_model_*.pth"):
                old.unlink()
            atomic_save(model.state_dict(), output_dir / f"best_model_{epoch:03d}.pth")
        else:
            bad_epochs += 1
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, ["epoch", "train_loss", "valid_loss", "lr", "best", "bad_epochs"])
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({"epoch": epoch, "train_loss": train_loss,
                             "valid_loss": valid_loss, "lr": optimizer.param_groups[0]["lr"],
                             "best": improved, "bad_epochs": bad_epochs})
        atomic_save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best_loss": best_loss,
            "bad_epochs": bad_epochs, "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "pending_gradients": {name: parameter.grad.detach().cpu()
                                  if parameter.grad is not None else None
                                  for name, parameter in model.named_parameters()},
        }, last_path)
        print(f"epoch={epoch} train={train_loss:.6f} valid={valid_loss:.6f} "
              f"lr={optimizer.param_groups[0]['lr']:.3g} best={improved}", flush=True)
        if bad_epochs >= 25:
            print(f"Early stop at epoch {epoch}; best valid loss={best_loss:.6f}")
            break
    print(f"Checkpoints and protocol: {output_dir}")


if __name__ == "__main__":
    main()
