"""Evaluate paired UCSF-BMSR Baseline and Full checkpoints on the fixed test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from data.ucsf_bmsr import UCSFPreparedDataset, get_ucsf_dataloader, read_manifest
from advanced_metrics import hd95_single
from wt_lesion_stratified import match_wt_components
from models.resunet3d import ResUNet3d
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary


def checkpoint_epoch(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def find_best(directory: Path) -> Path:
    checkpoints = list(directory.glob("best_model_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No best_model_*.pth in {directory}")
    return max(checkpoints, key=checkpoint_epoch)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_provenance(checkpoint: Path, manifest: Path, seed: int) -> dict:
    """Validate scratch-run provenance without changing any training files."""
    result = {"checkpoint": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint)}
    marker = checkpoint.parent / "initialization_protocol.json"
    if marker.is_file():
        initialization = json.loads(marker.read_text(encoding="utf-8"))
        if initialization.get("manifest_sha256") != file_sha256(manifest):
            raise ValueError("Evaluation manifest differs from this Full run's training manifest")
        if initialization.get("seed") != seed:
            raise ValueError("Evaluation seed differs from this Full run's training seed")
        completion = checkpoint.parent / "training_complete.json"
        if not completion.is_file() or json.loads(completion.read_text(encoding="utf-8")).get("status") != "completed":
            raise ValueError("Scratch Full is not marked completed; wait for training to finish")
        result["initialization"] = initialization
    else:
        result["initialization"] = "legacy run; no initialization protocol file"
    return result


def build_model(model_name: str):
    if model_name == "baseline":
        return ResUNet3d(in_channels=4, n_classes=1, n_channels=24)
    if model_name == "full":
        return ResUNetHFConcatBoundary(
            in_channels=4,
            n_classes=1,
            n_channels=24,
            fusion="concat",
            multiscale_context_v2=True,
        )
    raise ValueError(model_name)


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def safe_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def safe_std(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.std(finite)) if finite else float("nan")


def derive_volume_strata(volumes_mm3: list[float]) -> dict[str, tuple[float, float | None]]:
    """Fit approximately balanced lesion-volume strata without test data."""
    values, counts = np.unique(np.asarray(volumes_mm3, dtype=np.float64), return_counts=True)
    if len(values) < 3:
        raise ValueError("At least three distinct training lesion volumes are required")
    cumulative = np.cumsum(counts)
    first = int(np.argmin(np.abs(cumulative[:-2] - cumulative[-1] / 3.0)))
    candidates = np.arange(first + 1, len(values) - 1)
    second = int(candidates[np.argmin(np.abs(cumulative[candidates] - 2.0 * cumulative[-1] / 3.0))])
    return {
        "small": (0.0, float(values[first])),
        "medium": (float(values[first]), float(values[second])),
        "large": (float(values[second]), None),
    }


def classify_volume(
    volume_mm3: float, strata: dict[str, tuple[float, float | None]],
) -> str:
    if volume_mm3 <= strata["small"][1]:
        return "small"
    if volume_mm3 <= strata["medium"][1]:
        return "medium"
    return "large"


def collect_training_lesions(dataloader, min_component_size: int) -> list[dict]:
    """Measure GT components from the fixed training split in physical units."""
    rows = []
    for batch in dataloader:
        subject_id = str(batch["Id"][0])
        target = batch["mask"][0, 0].numpy() > 0.5
        spacing = tuple(float(value) for value in batch["spacing_dhw"][0].numpy())
        matching = match_wt_components(
            np.zeros_like(target), target, min_component_size=min_component_size,
        )
        voxel_volume = float(np.prod(spacing))
        for component_index, component in enumerate(matching["gt_components"]):
            rows.append({
                "subject_id": subject_id,
                "gt_component_index": component_index,
                "gt_voxels": component["size"],
                "gt_volume_mm3": component["size"] * voxel_volume,
            })
    if not rows:
        raise ValueError("No training lesions survived component filtering")
    return rows


def evaluate_model(
    model_name: str,
    checkpoint: Path,
    dataloader,
    device: str,
    threshold: float,
    min_component_size: int,
    volume_strata: dict[str, tuple[float, float | None]],
    metric_profile: str = "legacy-ucsf",
):
    if metric_profile not in {"legacy-ucsf", "manuscript", "both"}:
        raise ValueError(f"Unknown metric profile: {metric_profile}")
    model = build_model(model_name).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        state = state.state_dict()
    model.load_state_dict(state, strict=True)
    model.eval()
    case_rows = []
    lesion_rows = []
    with torch.no_grad():
        for batch in dataloader:
            subject_id = str(batch["Id"][0])
            image = batch["image"].to(device)
            target = batch["mask"][0, 0].numpy() > 0.5
            spacing = tuple(float(value) for value in batch["spacing_dhw"][0].numpy())
            output = model(image)
            if isinstance(output, tuple):
                output = output[0]
            probability = torch.sigmoid(output)[0, 0].cpu().numpy()
            prediction = probability >= threshold
            matching = match_wt_components(
                prediction,
                target,
                min_component_size=min_component_size,
            )
            legacy_hd95 = (hd95_single(prediction, target, voxel_spacing=spacing)
                           if metric_profile in {"legacy-ucsf", "both"} else None)
            manuscript_hd95 = None
            if metric_profile in {"manuscript", "both"}:
                # Reuse the exact surface implementation used by official nnU-Net scoring.
                from scripts.eval_official_nnunet import surface_metrics
                manuscript_hd95, _ = surface_metrics(prediction, target, spacing, profile="manuscript")
            row = {
                "model": model_name,
                "subject_id": subject_id,
                "threshold": threshold,
                "dice": binary_dice(prediction, target),
                "hd95_mm": manuscript_hd95 if metric_profile == "manuscript" else legacy_hd95,
                "gt_empty": not bool(target.any()),
                "pred_empty": not bool(prediction.any()),
                "gt_lesions": matching["gt_lesions"],
                "pred_lesions": matching["pred_lesions"],
                "tp_lesions": matching["tp"],
                "fp_lesions": matching["fp"],
                "fn_lesions": matching["fn"],
            }
            if metric_profile == "both":
                row["hd95_mm_manuscript"] = manuscript_hd95
            case_rows.append(row)
            matches = {item["gt_index"]: item for item in matching["matches"]}
            voxel_volume = float(np.prod(spacing))
            for gt_index, component in enumerate(matching["gt_components"]):
                match = matches.get(gt_index)
                volume_mm3 = component["size"] * voxel_volume
                lesion_rows.append({
                    "model": model_name,
                    "subject_id": subject_id,
                    "gt_component_index": gt_index,
                    "gt_voxels": component["size"],
                    "gt_volume_mm3": volume_mm3,
                    "volume_stratum": classify_volume(volume_mm3, volume_strata),
                    "detected": match is not None,
                    "matched_dice": float(match["dice"]) if match is not None else float("nan"),
                    "gt_anchored_dice": float(match["dice"]) if match is not None else 0.0,
                })
            print(f"Evaluated {model_name} {len(case_rows)}/{len(dataloader.dataset)}: {subject_id}", flush=True)
    return case_rows, lesion_rows


def summarize_lesion_strata(lesion_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, stratum), subset in lesion_frame.groupby(
        ["model", "volume_stratum"], sort=False,
    ):
        detected = subset["detected"].astype(bool)
        rows.append({
            "model": model_name,
            "volume_stratum": stratum,
            "gt_lesions": len(subset),
            "detected_lesions": int(detected.sum()),
            "missed_lesions": int((~detected).sum()),
            "lesion_recall": float(detected.mean()),
            "matched_lesion_dice": safe_mean(subset.loc[detected, "matched_dice"].tolist()),
            "gt_anchored_lesion_dice": safe_mean(subset["gt_anchored_dice"].tolist()),
        })
    order = {"small": 0, "medium": 1, "large": 2}
    result = pd.DataFrame(rows)
    result["_order"] = result["volume_stratum"].map(order)
    return result.sort_values(["model", "_order"]).drop(columns="_order")


def summarize_core(
    model_name: str, checkpoint: Path, cases: list[dict], lesions: list[dict],
) -> dict:
    """Return the compact UCSF endpoints corresponding to the final BraTS analysis."""
    tp = sum(row["tp_lesions"] for row in cases)
    fp = sum(row["fp_lesions"] for row in cases)
    fn = sum(row["fn_lesions"] for row in cases)
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    lesion_f1 = (
        2 * recall * precision / (recall + precision)
        if recall + precision else 0.0
    )
    hd95 = [row["hd95_mm"] for row in cases]
    dice = [row["dice"] for row in cases]
    small = [row for row in lesions if row["volume_stratum"] == "small"]
    return {
        "model": model_name,
        "checkpoint": str(checkpoint.resolve()),
        "n_test_scans": len(cases),
        "dice_mean": safe_mean(dice),
        "dice_std": safe_std(dice),
        "hd95_mm_mean": safe_mean(hd95),
        "hd95_mm_std": safe_std(hd95),
        "lesion_gt_anchored_dice": safe_mean(
            [row["gt_anchored_dice"] for row in lesions]
        ),
        "small_lesion_gt_anchored_dice": safe_mean(
            [row["gt_anchored_dice"] for row in small]
        ),
        "lesion_f1": lesion_f1,
        "small_gt_lesions": len(small),
        "hd95_finite_cases": sum(bool(np.isfinite(value)) for value in hd95),
        "gt_empty_cases": sum(bool(row.get("gt_empty", False)) for row in cases),
        "pred_empty_cases": sum(bool(row.get("pred_empty", False)) for row in cases),
        "tp_lesions": tp,
        "fp_lesions": fp,
        "fn_lesions": fn,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("/root/autodl-tmp/ucsf_bmsr_prepared/manifest.csv"),
    )
    parser.add_argument(
        "--training-root", type=Path,
        default=Path("/root/autodl-tmp/ucsf_baseline_full"),
    )
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--models", nargs="+", choices=("baseline", "full"),
                        default=("baseline", "full"), help="Select Full alone for a scratch-only training directory")
    parser.add_argument("--metric-profile", choices=("legacy-ucsf", "manuscript", "both"),
                        default="legacy-ucsf",
                        help="both performs inference once and writes historical plus harmonized HD95 summaries")
    parser.add_argument("--threshold", type=float, default=0.33)
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument(
        "--boundary-tolerance-mm", type=float, default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if len(args.models) != len(set(args.models)):
        parser.error("Each model may be selected only once")
    if not 0 < args.threshold < 1 or args.min_component_size < 1:
        parser.error("Threshold must be in (0,1) and minimum component size must be positive")

    frame = read_manifest(args.manifest)
    checkpoints = {name: find_best(args.training_root / f"seed{args.seed}" / name) for name in args.models}
    provenance = {name: checkpoint_provenance(path, args.manifest, args.seed)
                  for name, path in checkpoints.items()}
    protocol = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "seed": args.seed,
        "models": list(args.models),
        "checkpoints": provenance,
        "metric_profile": args.metric_profile,
        "summary_csv_profile": "manuscript" if args.metric_profile == "manuscript" else "legacy-ucsf",
        "threshold": args.threshold,
        "min_component_size": args.min_component_size,
        "split": {split: frame.loc[frame["split"] == split, "subject_id"].astype(str).tolist()
                  for split in ("train", "valid", "test")},
        "std_definition": "population standard deviation across test scans, not across training seeds",
        "hd95_profiles": {
            "legacy-ucsf": "maximum of the two directional 95th percentiles; NaN for an empty mask, excluded from the HD95 mean",
            "manuscript": "95th percentile of pooled bidirectional distances; both-empty 0, one-empty physical image diagonal",
        },
    }

    output_dir = args.output_dir or args.training_root / f"seed{args.seed}" / "evaluation"
    metadata_path = output_dir / "evaluation.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not metadata_path.is_file():
            raise ValueError("Nonempty evaluation directory has no protocol record; choose a new --output-dir")
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("protocol") != protocol:
            raise ValueError("Existing evaluation belongs to another checkpoint/protocol; choose a new --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_loader = get_ucsf_dataloader(
        UCSFPreparedDataset,
        str(args.manifest),
        "train",
        batch_size=1,
        num_workers=0,
    )
    training_lesions = collect_training_lesions(
        training_loader, args.min_component_size,
    )
    volume_strata = derive_volume_strata(
        [row["gt_volume_mm3"] for row in training_lesions]
    )
    for row in training_lesions:
        row["volume_stratum"] = classify_volume(row["gt_volume_mm3"], volume_strata)
    pd.DataFrame(training_lesions).to_csv(
        output_dir / "training_lesion_size_distribution.csv", index=False,
    )
    dataloader = get_ucsf_dataloader(
        UCSFPreparedDataset,
        str(args.manifest),
        "test",
        batch_size=1,
        num_workers=0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_cases = []
    all_lesions = []
    summaries = []
    manuscript_summaries = []
    for model_name in args.models:
        checkpoint = checkpoints[model_name]
        cases, lesions = evaluate_model(
            model_name,
            checkpoint,
            dataloader,
            device,
            args.threshold,
            args.min_component_size,
            volume_strata,
            metric_profile=args.metric_profile,
        )
        all_cases.extend(cases)
        all_lesions.extend(lesions)
        summaries.append(summarize_core(model_name, checkpoint, cases, lesions))
        if args.metric_profile == "both":
            common_cases = [{**row, "hd95_mm": row["hd95_mm_manuscript"]} for row in cases]
            manuscript_summaries.append(summarize_core(model_name, checkpoint, common_cases, lesions))
        if file_sha256(checkpoint) != provenance[model_name]["sha256"]:
            raise RuntimeError("Checkpoint changed during inference; use a stable completed model and a new output directory")

    case_frame = pd.DataFrame(all_cases)
    lesion_frame = pd.DataFrame(all_lesions)
    summary_frame = pd.DataFrame(summaries)
    lesion_strata_frame = summarize_lesion_strata(lesion_frame)
    case_frame.to_csv(output_dir / "per_case.csv", index=False)
    lesion_frame.to_csv(output_dir / "per_gt_lesion.csv", index=False)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(summary_frame.to_json(orient="records", indent=2), encoding="utf-8")
    if manuscript_summaries:
        common_frame = pd.DataFrame(manuscript_summaries)
        common_frame.to_csv(output_dir / "summary_manuscript.csv", index=False)
        (output_dir / "summary_manuscript.json").write_text(common_frame.to_json(orient="records", indent=2), encoding="utf-8")
    lesion_strata_frame.to_csv(output_dir / "summary_by_lesion_size.csv", index=False)
    if {"baseline", "full"}.issubset(args.models):
        paired = case_frame.pivot(index="subject_id", columns="model", values="dice")
        paired["full_minus_baseline_dice"] = paired["full"] - paired["baseline"]
        paired.reset_index().to_csv(output_dir / "paired_case_dice.csv", index=False)
    metadata = {
        "protocol": protocol,
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "threshold": args.threshold,
        "min_component_size": args.min_component_size,
        "lesion_size_definition": "individual GT connected-component physical volume; thresholds fitted on training split only",
        "lesion_volume_strata_mm3": volume_strata,
        "notes": [
            "The fixed test split is evaluated once; no five-fold averaging.",
            "A GT lesion is detected when one retained predicted component overlaps it after one-to-one Dice matching.",
            "Missed GT lesions receive zero GT-anchored Dice.",
            "Small/medium/large thresholds are never fitted on validation or test data.",
            "The primary table is restricted to Dice, HD95, overall GT-anchored lesion Dice, small-lesion GT-anchored Dice, and lesion F1.",
        ],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    if manuscript_summaries:
        print("\nHarmonized manuscript HD95 (same predictions; no second inference):")
        print(pd.DataFrame(manuscript_summaries).to_string(index=False))
    archive = output_dir / "results_for_table.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(output_dir.iterdir()):
            if path.suffix in {".json", ".csv"}:
                stream.write(path, path.name)
        for name, checkpoint in checkpoints.items():
            for filename in ("initialization_protocol.json", "training_complete.json", "train_log.csv"):
                path = checkpoint.parent / filename
                if path.is_file():
                    stream.write(path, f"training/{name}/{filename}")
    print(f"Download for table: {archive.resolve()}")
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
