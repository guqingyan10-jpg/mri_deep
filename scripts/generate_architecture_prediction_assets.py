"""Export real model outputs for the AFBMS-ResUNet architecture figure.

The script follows the project's BraTS test preprocessing and loads the Full V2
checkpoint strictly. It exports MRI slices, segmentation-head predictions,
boundary-head predictions, the high-frequency residual, raw arrays, and a JSON
provenance record. It never labels GT or an eroded GT target as a prediction.

Example (run from the repository root on AutoDL)::

    python scripts/generate_architecture_prediction_assets.py \
      --case-dir /root/autodl-tmp/MICCAI_BraTS2020_TrainingData/BraTS20_Training_309 \
      --checkpoint /root/autodl-tmp/mri_deep/deliverables/AFBMS_ResUNet_handoff_20260907/artifacts/weights/afbms_resunet/seed_55/best_model_epoch_090.pth \
      --out outputs/architecture_predictions

The checkpoint must belong to ``ResUNetHFConcatBoundary`` with concat HF fusion
and ``multiscale_context_v2=True``. Use ``--n-classes 1`` for a binary model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", type=Path, required=True,
                   help="Directory containing CASE_flair/t1/t1ce/t2/seg.nii(.gz).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/architecture_predictions"))
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--n-classes", type=int, default=3, choices=(1, 3))
    p.add_argument("--threshold", type=float, default=0.33,
                   help="Display threshold for sigmoid probabilities (default: 0.33).")
    p.add_argument("--slice-index", type=int, default=None,
                   help="Preprocessed Z slice. Default: largest GT ET/foreground slice.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--no-panel", action="store_true",
                   help="Do not create the small contact panel PNG.")
    return p.parse_args()


def find_case_file(case_dir: Path, case: str, suffix: str) -> Path:
    for ext in (".nii", ".nii.gz"):
        candidate = case_dir / f"{case}_{suffix}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {case}_{suffix}.nii(.gz) in {case_dir}")


def load_normalized_volume(path: Path, crop: bool = True) -> np.ndarray:
    nii = nib.load(str(path), mmap=False)
    arr = np.asarray(nii.dataobj, dtype=np.float32)
    if crop:
        arr = arr[40:210, 40:210, 20:120]
    if arr.ndim != 3 or not np.isfinite(arr).all():
        raise ValueError(f"Invalid volume: {path} shape={arr.shape}")
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        raise ValueError(f"Constant volume cannot be normalized: {path}")
    return (arr - lo) / (hi - lo)


def load_model(repo_root: Path, checkpoint: Path, n_classes: int, device: torch.device):
    sys.path.insert(0, str(repo_root.resolve()))
    from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary

    model = ResUNetHFConcatBoundary(
        in_channels=4,
        n_classes=n_classes,
        n_channels=24,
        fusion="concat",
        multiscale_context_v2=True,
    )
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before the weights_only keyword
        state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model_state_dict", state))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    clean = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        # Older saved checkpoints wrapped the 1x1 output conv in Sequential.
        key = key.replace("out.conv.0.", "out.conv.")
        clean[key] = value
    model.load_state_dict(clean, strict=True)
    return model.to(device).eval()


def save_slice(array: np.ndarray, path: Path, signed: bool = False) -> None:
    view = np.rot90(np.asarray(array))
    if signed:
        limit = float(np.quantile(np.abs(view), 0.995)) or 1.0
        view = np.clip((view / (2.0 * limit)) + 0.5, 0.0, 1.0)
    view = np.clip(view * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(view).save(path)


def save_panel(out: Path, selected: int, overlay: np.ndarray, boundary: np.ndarray,
               high: np.ndarray) -> None:
    panels = [
        ("Segmentation prediction", np.rot90(overlay)),
        ("Boundary-head prediction", np.rot90(boundary)),
        ("Signed high-frequency residual", np.rot90(high, axes=(0, 1))),
    ]
    width, height = 300, 340
    canvas = Image.new("RGB", (width * len(panels), height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, image) in enumerate(panels):
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        im = Image.fromarray(image).resize((width - 20, width - 20))
        canvas.paste(im, (i * width + 10, 28))
        draw.text((i * width + 10, 8), title, fill="black")
    canvas.save(out / f"architecture_prediction_panel_z{selected:03d}.png")


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    case_dir = args.case_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    out = args.out.resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(case_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    case = case_dir.name
    out.mkdir(parents=True, exist_ok=True)

    suffixes = ("flair", "t1", "t1ce", "t2")
    modality_names = ("FLAIR", "T1", "T1ce", "T2")
    volumes = [load_normalized_volume(find_case_file(case_dir, case, s)) for s in suffixes]
    if len({v.shape for v in volumes}) != 1:
        raise ValueError(f"Modalities do not share a shape: {[v.shape for v in volumes]}")
    image = np.moveaxis(np.stack(volumes), (0, 1, 2, 3), (0, 3, 2, 1)).astype(np.float32)

    raw_gt = np.asarray(nib.load(str(find_case_file(case_dir, case, "seg")), mmap=False).dataobj)
    raw_gt = raw_gt[40:210, 40:210, 20:120]
    if args.n_classes == 1:
        gt = np.moveaxis(np.stack([raw_gt > 0]), (0, 1, 2, 3), (0, 3, 2, 1)).astype(np.float32)
    else:
        gt = np.moveaxis(np.stack([
            raw_gt > 0,
            (raw_gt == 1) | (raw_gt == 4),
            raw_gt == 4,
        ]), (0, 1, 2, 3), (0, 3, 2, 1)).astype(np.float32)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_model(args.repo_root.resolve(), checkpoint, args.n_classes, device)
    x = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        seg_logits, boundary_logits = model(x)
        seg_prob = seg_logits.sigmoid()[0].cpu().numpy()
        boundary_prob = boundary_logits.sigmoid()[0].cpu().numpy()
        high = model.edge_extractor(x)[0].cpu().numpy()

    pred = seg_prob >= args.threshold
    boundary_pred = boundary_prob >= args.threshold
    if args.slice_index is None:
        areas = gt.sum(axis=(0, 2, 3)) if args.n_classes == 1 else gt[2].sum(axis=(1, 2))
        selected = int(np.argmax(areas))
    else:
        selected = int(args.slice_index)
    if not 0 <= selected < image.shape[1]:
        raise IndexError(f"slice-index {selected} outside [0, {image.shape[1] - 1}]")

    for channel, name in enumerate(modality_names):
        save_slice(image[channel, selected], out / f"mri_{name}.png")
    gray = np.repeat(image[2, selected, ..., None], 3, axis=-1)
    overlay = gray.copy()
    mask_rgb = np.zeros_like(gray)
    colors = np.array([[1.0, 0.78, 0.0], [0.15, 0.80, 0.30], [0.93, 0.12, 0.13]])
    for channel in range(args.n_classes):
        area = pred[channel, selected]
        color = colors[channel if args.n_classes == 3 else 2]
        overlay[area] = 0.38 * gray[area] + 0.62 * color
        mask_rgb[area] = color
    boundary_slice = boundary_pred[:, selected].any(axis=0)
    save_slice(overlay, out / "seg_prediction_overlay.png")
    save_slice(mask_rgb, out / "seg_prediction_mask.png")
    save_slice(boundary_slice.astype(np.float32), out / "boundary_head_mask.png")
    save_slice(boundary_prob[:, selected].max(axis=0), out / "boundary_head_probability.png")
    save_slice(high[2, selected], out / "hf_signed.png", signed=True)
    if not args.no_panel:
        save_panel(out, selected, overlay, boundary_slice.astype(np.float32), high[2, selected])

    np.savez_compressed(
        out / "prediction_arrays.npz",
        image=image,
        gt=gt,
        seg_prob=seg_prob,
        boundary_prob=boundary_prob,
        selected_slice=np.int64(selected),
    )
    provenance = {
        "case": case,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model": "ResUNetHFConcatBoundary(n_channels=24, fusion=concat, multiscale_context_v2=True)",
        "strict_checkpoint_load": True,
        "device": str(device),
        "modalities_in_model_order": list(modality_names),
        "preprocessing": "Exact BraTS test path: crop [40:210,40:210,20:120], per-modality min-max normalization, axis order C,Z,Y,X; no resize or augmentation.",
        "slice_index": selected,
        "slice_selection": "Maximum GT ET area for BraTS, or maximum foreground area for binary mode; selection is for an architecture illustration.",
        "segmentation_threshold": args.threshold,
        "boundary_threshold": args.threshold,
        "prediction_sources": {
            "segmentation": "segmentation head logits followed by sigmoid and threshold",
            "boundary": "auxiliary boundary head logits followed by sigmoid and threshold",
            "gt_is_not_prediction": True,
        },
        "display_priority": "ET red > TC green > WT yellow for the nested BraTS channels",
        "learned_alpha": float(model.multiscale_context.alpha.detach().cpu()),
    }
    (out / "prediction_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for asset in sorted(out.iterdir()):
            if asset.is_file():
                zf.write(asset, arcname=f"{out.name}/{asset.name}")
    print(json.dumps({
        "output": str(out),
        "archive": str(archive),
        "case": case,
        "slice": selected,
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
