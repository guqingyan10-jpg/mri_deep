"""Compare five requested models on training-defined lesion-size strata.

Metrics are lesion-level (not case-level): one-to-one matched lesion recall,
miss rate, matched lesion Dice, and GT-anchored lesion Dice.  Size thresholds
are fitted on the training split only and then frozen for validation/test.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
from torch.utils.data import ConcatDataset, DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BratsDataset, get_dataloader
from models.resunet_edge import ResUNetEdge
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
from models.resunet3d import ResUNet3d
from evaluation.wt_lesion_stratified import (
    classify_lesion_size,
    derive_size_strata,
    match_lesion_components,
    summarize_stratified_cases,
)


# Keep this registry independent from scripts/eval_all_experiments.py: this
# script is a separate WT lesion-level analysis of the five requested runs.
# The values are deliberately literal so the registry can be audited without
# importing PyTorch (see tests/test_wt_eval_model_registry.py).
MODEL_SPECS = (
    {
        "name": "baseline",
        "label": "ResUNet (BCE–Dice)",
        "model_type": "resunet",
        "checkpoint_dir": "/root/autodl-tmp/ResUNet_model",
        "model_kwargs": {"in_channels": 4, "n_classes": 3, "n_channels": 24},
        "key_remap": None,
    },
    {
        "name": "edge_laplacian",
        "label": "ResUNet + LHFC",
        "model_type": "edge",
        "checkpoint_dir": "/root/autodl-tmp/ResUNet_Edge_concat_laplacian_model",
        "model_kwargs": {
            "in_channels": 4,
            "n_classes": 3,
            "n_channels": 24,
            "fusion": "concat",
            "edge_type": "laplacian",
        },
        "key_remap": "edge",
    },
    {
        "name": "hf_w01",
        "label": "ResUNet + LHFC + ABS (λ_b = 0.1)",
        "model_type": "hf",
        "checkpoint_dir": "/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.1_model",
        "model_kwargs": {"in_channels": 4, "n_classes": 3, "n_channels": 24},
        "key_remap": None,
    },
    {
        "name": "hf_w01_multiscale_v2",
        "label": "AFBMS-ResUNet",
        "model_type": "hf",
        "checkpoint_dir": "/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model",
        "model_kwargs": {
            "in_channels": 4,
            "n_classes": 3,
            "n_channels": 24,
            "multiscale_context_v2": True,
        },
        "key_remap": None,
    },
    {
        "name": "hf_w005",
        "label": "ResUNet + LHFC + ABS (λ_b = 0.05)",
        "model_type": "hf",
        "checkpoint_dir": "/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.05_model",
        "model_kwargs": {"in_channels": 4, "n_classes": 3, "n_channels": 24},
        "key_remap": None,
    },
)

MODEL_CLASSES = {
    "resunet": ResUNet3d,
    "edge": ResUNetEdge,
    "hf": ResUNetHFConcatBoundary,
}
STRUCTURE_26 = ndimage.generate_binary_structure(3, 3)


def evaluation_dataloader(csv_path, phase):
    """Build a deterministic evaluation loader and its case manifest.

    ``valid_test`` is an exploratory pooled analysis: it concatenates the
    project's already-fixed validation and test datasets without resampling.
    The two original split labels are retained in the manifest so the pooled
    cohort can be audited.  Training cases are never included here.
    """
    split_names = ("valid", "test") if phase == "valid_test" else (phase,)
    split_loaders = [
        get_dataloader(BratsDataset, csv_path, phase=name, batch_size=1)
        for name in split_names
    ]
    manifest_rows = []
    for name, loader in zip(split_names, split_loaders):
        manifest_rows.extend(
            {"split": name, "case_id": case_id}
            for case_id in loader.dataset.df["Brats20ID"].tolist()
        )
    case_ids = [row["case_id"] for row in manifest_rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation splits contain duplicate case IDs")

    if len(split_loaders) == 1:
        return split_loaders[0], pd.DataFrame(manifest_rows)
    combined_dataset = ConcatDataset([loader.dataset for loader in split_loaders])
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )
    return combined_loader, pd.DataFrame(manifest_rows)


def find_checkpoint(directory: str) -> str | None:
    path = Path(directory)
    if not path.exists():
        return None
    candidates = list(path.glob("best_model_*.pth"))
    if not candidates:
        candidates = list(path.glob("last_epoch_model_*.pth"))
    if not candidates:
        return None

    def epoch_number(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    return str(sorted(candidates, key=epoch_number)[-1])


def load_model(spec, checkpoint: str, device):
    model_class = MODEL_CLASSES[spec["model_type"]]
    model = model_class(**spec["model_kwargs"]).to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {}
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        key = key.replace("out.conv.0.", "out.conv.")
        if spec.get("key_remap") == "edge" and key.startswith("sobel."):
            # Early Edge checkpoints named this fixed extractor ``sobel``
            # even when the run used the Laplacian implementation.
            key = key.replace("sobel.", "edge_extractor.", 1)
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(
            f"  [WARN] checkpoint compatibility: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    model.eval()
    return model


def _safe_json(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    return value


def training_component_sizes(csv_path, channel, min_size):
    """Extract components from the fixed training split without model inference."""
    dataset = get_dataloader(
        BratsDataset, csv_path, phase="train", batch_size=1
    ).dataset
    sizes = []
    for row in tqdm(dataset.df.itertuples(index=False), total=len(dataset.df),
                    desc="Fit training lesion strata"):
        case_id = row.Brats20ID
        mask = dataset.load_img(os.path.join(row.path, case_id + "_seg.nii"))
        if dataset.is_resize:
            mask = dataset.resize(mask)
        mask = dataset.preprocess_mask_labels(mask)[channel]
        labeled, count = ndimage.label(mask > 0, structure=STRUCTURE_26)
        counts = np.bincount(labeled.ravel())[1 : count + 1]
        sizes.extend(int(value) for value in counts if value >= min_size)
    return sizes


def evaluate_model(model, dataloader, device, threshold, min_size, channel, strata, region):
    case_results = []
    detail_rows = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"{region} lesion evaluation"):
            images = data["image"].to(device)
            targets = data["mask"].cpu().numpy()
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            predictions = (torch.sigmoid(logits) >= threshold).cpu().numpy()
            ids = data.get("Id", ["unknown"] * len(images))
            if not isinstance(ids, list):
                ids = [ids]

            for i, case_id in enumerate(ids):
                result = match_lesion_components(
                    predictions[i, channel],
                    targets[i, channel],
                    structure=STRUCTURE_26,
                    min_component_size=min_size,
                )
                result["case_id"] = case_id
                case_results.append(result)

                matched_by_gt = {m["gt_index"]: m for m in result["matches"]}
                for gi, component in enumerate(result["gt_components"]):
                    stratum = classify_lesion_size(component["size"], strata)
                    match = matched_by_gt.get(gi)
                    detail_rows.append(
                        {
                            "case_id": case_id,
                            "gt_lesion_id": component["id"],
                            "stratum": stratum,
                            "gt_voxels": component["size"],
                            "pred_lesion_id": match["pred_id"] if match else "",
                            "pred_voxels": match["pred_size"] if match else "",
                            "intersection_voxels": match["intersection"] if match else 0,
                            "matched_dice": match["dice"] if match else np.nan,
                            "detected": bool(match),
                            "gt_anchored_dice": match["dice"] if match else 0.0,
                        }
                    )
    return case_results, detail_rows


def flatten_summaries(model_name, summaries):
    rows = []
    for stratum, summary in summaries.items():
        row = {"model": model_name, **summary}
        row.pop("matched_dice_values", None)
        rows.append(row)
    return rows


def plot_summary(summary_df: pd.DataFrame, output: Path, region: str):
    models = list(summary_df["model"].drop_duplicates())
    strata = ["small", "medium", "large"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    metrics = [
        ("lesion_recall", "Lesion recall", "Recall"),
        ("matched_lesion_dice", "Matched lesion Dice", "Dice"),
        ("miss_rate", "Miss rate", "Miss rate"),
    ]
    x = np.arange(len(strata))
    width = 0.18
    colors = [
        "#2C7FB8",  # Baseline
        "#41AB5D",  # Edge (Laplacian, concat)
        "#D95F0E",  # HF Concat Boundary w=0.1
        "#756BB1",  # HF Concat Boundary + Multi-scale V2
        "#E7298A",  # HF Concat Boundary w=0.05
    ]
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for mi, model in enumerate(models):
            values = []
            for stratum in strata:
                match = summary_df[
                    (summary_df["model"] == model)
                    & (summary_df["stratum"] == stratum)
                ]
                values.append(float(match.iloc[0][metric]) if len(match) else np.nan)
            ax.bar(x + (mi - 1.5) * width, values, width, label=model, color=colors[mi])
        ax.set_title(title)
        ax.set_xticks(x, [s.capitalize() for s in strata])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{region} lesion-level performance by ground-truth lesion size", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main(default_region="WT"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="tumourCSV.csv")
    parser.add_argument("--region", choices=("WT", "ET"), default=default_region)
    parser.add_argument(
        "--phase",
        choices=("valid", "test", "valid_test"),
        default="test",
        help=(
            "Evaluation cohort. valid_test pools the fixed validation and test "
            "cases for exploratory analysis; it does not refit lesion strata."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.33)
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument(
        "--strata-json",
        default=None,
        help="Frozen training-split strata JSON from derive_train_lesion_strata.py",
    )
    parser.add_argument("--device", default=None, help="e.g. cuda or cpu")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--edge-laplacian-checkpoint", default=None)
    parser.add_argument("--hf-w01-checkpoint", dest="hf_w01_checkpoint", default=None)
    parser.add_argument(
        "--hf-w01-multiscale-v2-checkpoint",
        dest="hf_w01_multiscale_v2_checkpoint",
        default=None,
    )
    parser.add_argument("--hf-w005-checkpoint", dest="hf_w005_checkpoint", default=None)
    args = parser.parse_args()

    region = args.region
    channel = {"WT": 0, "ET": 2}[region]
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.phase == "valid_test":
        output_dir = Path(
            f"{region.lower()}_lesion_stratified_valid_test_results"
        )
    else:
        output_dir = Path(f"{region.lower()}_lesion_stratified_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_overrides = {
        "baseline": args.baseline_checkpoint,
        "edge_laplacian": args.edge_laplacian_checkpoint,
        "hf_w01": args.hf_w01_checkpoint,
        "hf_w01_multiscale_v2": args.hf_w01_multiscale_v2_checkpoint,
        "hf_w005": args.hf_w005_checkpoint,
    }

    training_sizes = None
    distribution = None
    if args.strata_json:
        strata_path = Path(args.strata_json)
        with open(strata_path, encoding="utf-8") as handle:
            strata_metadata = json.load(handle)
        if strata_metadata.get("region") != region:
            raise ValueError(
                f"strata region {strata_metadata.get('region')!r} does not match {region}"
            )
        if strata_metadata.get("fit_split") != "train":
            raise ValueError("strata JSON was not fitted on the training split")
        if strata_metadata.get("connectivity") != 26:
            raise ValueError("strata JSON must use 26-connectivity")
        if strata_metadata.get("min_component_size") != args.min_component_size:
            raise ValueError("strata JSON minimum component size does not match CLI")
        strata = OrderedDict(
            (name, tuple(strata_metadata["strata"][name]))
            for name in ("small", "medium", "large")
        )
        distribution_path = strata_path.with_name(
            f"{region.lower()}_training_lesion_size_distribution.csv"
        )
        if distribution_path.exists():
            distribution = pd.read_csv(distribution_path)
    else:
        training_sizes = training_component_sizes(
            args.csv, channel, args.min_component_size
        )
        strata = derive_size_strata(training_sizes, args.min_component_size)
        strata_metadata = {
            "region": region,
            "fit_split": "train",
            "split_random_state": 10,
            "training_cases": None,
            "connectivity": 26,
            "min_component_size": args.min_component_size,
            "training_component_count": len(training_sizes),
            "strata": strata,
            "stratum_counts": {
                name: sum(
                    value >= lower and (upper is None or value <= upper)
                    for value in training_sizes
                )
                for name, (lower, upper) in strata.items()
            },
        }
    dataloader, case_manifest = evaluation_dataloader(args.csv, args.phase)
    print(f"Device: {device}; {args.phase} cases: {len(dataloader.dataset)}")
    print(
        "Evaluation split counts:",
        ", ".join(
            f"{name}={count}"
            for name, count in case_manifest["split"].value_counts(sort=False).items()
        ),
    )
    print(f"Region: {region} (channel {channel}); connectivity: 26; minimum component size:", args.min_component_size)
    print("Training-defined strata:", ", ".join(f"{name}={lo}-{hi or 'inf'}" for name, (lo, hi) in strata.items()))
    strata_metadata["apply_split"] = args.phase
    strata_metadata["evaluation_splits"] = case_manifest["split"].drop_duplicates().tolist()
    strata_metadata["evaluation_cases"] = len(case_manifest)
    strata_metadata["train_fraction"] = 0.7
    with open(output_dir / f"{region.lower()}_lesion_strata.json", "w", encoding="utf-8") as handle:
        json.dump(_safe_json(strata_metadata), handle, indent=2)
    if distribution is None and training_sizes is not None:
        distribution = (
            pd.Series(training_sizes, name="lesion_voxels")
            .value_counts()
            .rename_axis("lesion_voxels")
            .rename("lesion_count")
            .sort_index()
            .reset_index()
        )
        distribution["cumulative_count"] = distribution["lesion_count"].cumsum()
        distribution["cumulative_fraction"] = (
            distribution["cumulative_count"] / len(training_sizes)
        )
        distribution["stratum"] = distribution["lesion_voxels"].map(
            lambda value: classify_lesion_size(value, strata)
        )
    if distribution is not None:
        distribution.to_csv(
            output_dir / f"{region.lower()}_training_lesion_size_distribution.csv",
            index=False,
        )

    all_rows = []
    all_details = []
    all_json = {}
    for spec in MODEL_SPECS:
        model_name = spec["label"]
        checkpoint = checkpoint_overrides[spec["name"]] or find_checkpoint(
            spec["checkpoint_dir"]
        )
        if not checkpoint:
            print(
                f"[SKIP] {model_name}: checkpoint not found "
                f"({spec['checkpoint_dir']})"
            )
            continue
        print(f"\nEvaluating {model_name}: {checkpoint}")
        model = load_model(spec, checkpoint, device)
        case_results, detail_rows = evaluate_model(
            model, dataloader, device, args.threshold, args.min_component_size,
            channel, strata, region,
        )
        summaries = summarize_stratified_cases(case_results, strata)
        all_rows.extend(flatten_summaries(model_name, summaries))
        for row in detail_rows:
            row["model"] = model_name
            all_details.append(row)
        all_json[model_name] = summaries
        for stratum in ("small", "medium", "large"):
            s = summaries[stratum]
            print(
                f"  {stratum:<7} N={s['gt_lesions']:<4} "
                f"Recall={s['lesion_recall']:.4f} "
                f"MatchedDice={s['matched_lesion_dice']:.4f} "
                f"GTDice={s['gt_anchored_lesion_dice']:.4f} "
                f"Miss={s['miss_rate']:.4f}"
            )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(all_rows)
    detail_df = pd.DataFrame(all_details)
    prefix = region.lower()
    case_manifest.to_csv(
        output_dir / f"{prefix}_evaluated_cases.csv", index=False
    )
    summary_df.to_csv(output_dir / f"{prefix}_lesion_stratified_summary.csv", index=False)
    detail_df.to_csv(output_dir / f"{prefix}_lesion_stratified_detail.csv", index=False)
    with open(output_dir / f"{prefix}_lesion_stratified_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_safe_json(all_json), handle, indent=2)
    if not summary_df.empty:
        plot_summary(summary_df, output_dir / f"{prefix}_lesion_stratified_comparison.png", region)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
