"""Score the newly trained scratch BraTS Full model for the five-column table.

Reuses the existing Full architecture, best-checkpoint selector, surface
metrics, and 26-connected ET matcher. It never reads other models' weights
or supplies historical comparison values.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation")]

import numpy as np
import pandas as pd
import torch

from data.dataset import BratsDataset, get_dataloader
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
from scripts.eval_ucsf_baseline_full import find_best
from scripts.eval_official_nnunet import (
    _mean_std, binary_dice, file_sha256, json_safe, surface_metrics, summarize_lesions,
)
from advanced_metrics import hd95_single, nsd_single
from wt_lesion_stratified import classify_wt_size, match_wt_components


STRATA = {"small": (10, 44), "medium": (45, 4678), "large": (4679, None)}
MODEL_NAME = "AFBMS-ResUNet (from scratch)"


def initialization_record(checkpoint, csv_path, seed):
    marker = checkpoint.parent / "initialization_protocol.json"
    if not marker.is_file():
        raise ValueError("No initialization_protocol.json; select the certified scratch run")
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("seed") != seed or record.get("manifest_sha256") != file_sha256(csv_path):
        raise ValueError("Checkpoint seed or training CSV differs from this evaluation")
    if record.get("dataset") != "BraTS2020" or record.get("initialization") != "from_scratch":
        raise ValueError("Selected checkpoint is not a from-scratch BraTS Full run")
    settings = record.get("model_settings", {})
    if not (settings.get("multiscale_context_v2") is True and settings.get("boundary_weight") == 0.1
            and settings.get("fusion") == "concat" and settings.get("base_channels") == 24
            and not settings.get("multiscale_context") and not settings.get("multiscale_context_v3")):
        raise ValueError("Initialization record differs from the formal LHFC+ABS(0.1)+AR-MSC V2 model")
    return record


def load_full(checkpoint, device):
    model = ResUNetHFConcatBoundary(
        in_channels=4, n_classes=3, n_channels=24,
        fusion="concat", multiscale_context_v2=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def evaluate_loaded_model(model, dataloader, device, profile, tolerance):
    case_rows, lesion_rows = [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            target = batch["mask"].cpu().numpy() > 0.5
            segmentation, _ = model(images)
            if tuple(segmentation.shape) != tuple(target.shape) or segmentation.shape[1] != 3:
                raise ValueError("Expected three WT/TC/ET logits on the unchanged study crop")
            if not torch.isfinite(segmentation).all():
                raise ValueError("Model produced nonfinite logits")
            predictions = (torch.sigmoid(segmentation) >= 0.33).cpu().numpy()
            for index, case_id in enumerate(batch["Id"]):
                pred, gt = predictions[index], target[index]
                values = [binary_dice(p, t) for p, t in zip(pred, gt)]
                if profile == "legacy-brats":
                    hd = hd95_single(pred[2], gt[2], voxel_spacing=(1, 1, 1))
                    boundary = nsd_single(pred[2], gt[2], tau=tolerance, voxel_spacing=(1, 1, 1))
                else:
                    hd, boundary = surface_metrics(pred[2], gt[2], (1, 1, 1), tolerance, "manuscript")
                matching = match_wt_components(pred[2], gt[2], min_component_size=10)
                case_rows.append({"model": MODEL_NAME, "case_id": str(case_id),
                    **dict(zip(("WT_Dice", "TC_Dice", "ET_Dice"), values)),
                    "Macro_Dice": float(np.mean(values)), "ET_HD95": hd,
                    "Boundary_Dice": boundary, "gt_empty": not bool(gt[2].any()),
                    "pred_empty": not bool(pred[2].any()),
                    "gt_lesions": matching["gt_lesions"], "pred_lesions": matching["pred_lesions"],
                    "tp_lesions": matching["tp"], "fp_lesions": matching["fp"], "fn_lesions": matching["fn"]})
                matches = {pair["gt_index"]: pair for pair in matching["matches"]}
                for component_index, component in enumerate(matching["gt_components"]):
                    pair = matches.get(component_index)
                    lesion_rows.append({"model": MODEL_NAME, "case_id": str(case_id),
                        "gt_component_index": component_index, "gt_voxels": component["size"],
                        "stratum": classify_wt_size(component["size"], STRATA),
                        "detected": pair is not None,
                        "matched_dice": pair["dice"] if pair else float("nan"),
                        "gt_anchored_dice": pair["dice"] if pair else 0.0})
                print(f"Evaluated {len(case_rows)}/{len(dataloader.dataset)}: {case_id}", flush=True)
    return case_rows, lesion_rows


def summarize_model(checkpoint, seed, profile, cases, lesions):
    strata = summarize_lesions(cases, lesions)
    overall = strata[-1]
    result = {"model": MODEL_NAME, "model_id": "full_scratch", "seed": seed,
              "checkpoint": str(checkpoint), "n_test_scans": len(cases), "metric_profile": profile}
    for field in ("Macro_Dice", "WT_Dice", "TC_Dice", "ET_Dice", "ET_HD95", "Boundary_Dice"):
        result[field + "_mean"], result[field + "_std"] = _mean_std([row[field] for row in cases])
    result.update(Lesion_F1=overall["lesion_f1"],
                  tp_lesions=overall["detected_lesions"], fp_lesions=overall["fp_lesions"],
                  fn_lesions=overall["missed_lesions"],
                  ET_HD95_finite_cases=sum(bool(np.isfinite(row["ET_HD95"])) for row in cases),
                  Boundary_Dice_finite_cases=sum(bool(np.isfinite(row["Boundary_Dice"])) for row in cases),
                  gt_empty_cases=sum(row["gt_empty"] for row in cases),
                  pred_empty_cases=sum(row["pred_empty"] for row in cases))
    return result, strata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Scratch Full directory containing best_model_*.pth; default is brats/seed<seed>")
    parser.add_argument("--csv", type=Path, default=ROOT / "tumourCSV.csv")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-profile", choices=("legacy-brats", "manuscript"), default="legacy-brats")
    parser.add_argument("--boundary-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--dry-run", action="store_true", help="Verify paths and frozen IDs without reading MRI")
    args = parser.parse_args(argv)
    if args.boundary_tolerance_mm <= 0:
        parser.error("Boundary tolerance must be positive")
    directory = args.checkpoint_dir or Path("/root/autodl-tmp/afbms_full_from_scratch/brats") / f"seed{args.seed}"
    checkpoint = find_best(directory).resolve()
    initialization = initialization_record(checkpoint, args.csv, args.seed)
    loaders = {split: get_dataloader(BratsDataset, str(args.csv), phase=split, batch_size=1, num_workers=0)
               for split in ("train", "valid", "test")}
    split_ids = {split: loader.dataset.df["Brats20ID"].astype(str).tolist() for split, loader in loaders.items()}
    canonical = {split: get_dataloader(BratsDataset, str(ROOT / "tumourCSV.csv"), phase=split,
                                     batch_size=1, num_workers=0).dataset.df["Brats20ID"].astype(str).tolist()
                 for split in split_ids}
    if split_ids != canonical or {split: len(ids) for split, ids in split_ids.items()} != {"train": 257, "valid": 74, "test": 37}:
        raise ValueError("Case IDs/order differ from the frozen 257/74/37 study split")
    checkpoint_hash = file_sha256(checkpoint)
    protocol = {"dataset": "BraTS2020", "seed": args.seed, "model": MODEL_NAME,
                "csv_sha256": file_sha256(args.csv), "split": split_ids,
                "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                "initialization": initialization, "metric_profile": args.metric_profile,
                "region_order": ["WT", "TC", "ET"], "threshold": 0.33,
                "evaluation_spacing_dhw_mm": [1, 1, 1],
                "crop_xyz": [[40, 210], [40, 210], [20, 120]],
                "component_connectivity": 26, "min_component_size": 10,
                "boundary_tolerance_mm": args.boundary_tolerance_mm,
                "lesion_f1": "2*sum(TP)/(2*sum(TP)+sum(FP)+sum(FN))",
                "std_definition": "between test cases for this seed; not between random seeds",
                "empty_surface_rule": "NaN excluded from mean" if args.metric_profile == "legacy-brats" else "both empty HD95=0/BoundaryDice=1; one empty HD95=physical diagonal/BoundaryDice=0",
                "hd95_definition": "maximum of directional 95th percentiles" if args.metric_profile == "legacy-brats" else "95th percentile of pooled bidirectional distances"}
    print(f"Selected scratch Full: {checkpoint}", flush=True)
    if args.dry_run:
        print("Checkpoint, scratch initialization and frozen case split verified; no inference was run.")
        return
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    marker = args.output_dir / "evaluation_protocol.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Nonempty output directory belongs to another protocol; choose a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    cases, lesions = evaluate_loaded_model(load_full(checkpoint, device), loaders["test"], device,
                                          args.metric_profile, args.boundary_tolerance_mm)
    if [row["case_id"] for row in cases] != split_ids["test"]:
        raise ValueError("Inference did not produce exactly the frozen 37-case test set in order")
    counts = {s: sum(row["stratum"] == s for row in lesions) for s in STRATA}
    if counts != {"small": 31, "medium": 36, "large": 26}:
        raise ValueError(f"ET GT lesion inventory differs from frozen study: {counts}")
    if file_sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during evaluation; wait for training to complete and choose a new output directory")
    summary, strata = summarize_model(checkpoint, args.seed, args.metric_profile, cases, lesions)
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(cases).to_csv(args.output_dir / "per_case.csv", index=False)
    pd.DataFrame(lesions).to_csv(args.output_dir / "per_gt_lesion.csv", index=False)
    pd.DataFrame(strata).to_csv(args.output_dir / "summary_by_lesion_size.csv", index=False)
    columns = ["model", "Macro_Dice_mean", "ET_Dice_mean", "ET_HD95_mean", "Boundary_Dice_mean", "Lesion_F1"]
    pd.DataFrame([summary])[columns].to_csv(args.output_dir / "comparison_row.csv", index=False)
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False), flush=True)
    archive = args.output_dir / "results_for_table.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(args.output_dir.iterdir()):
            if path.suffix in {".csv", ".json"}:
                stream.write(path, path.name)
        for filename in ("initialization_protocol.json", "train_log.csv", "alpha_history.csv"):
            path = checkpoint.parent / filename
            if path.is_file():
                stream.write(path, f"training/full/{filename}")
    print(f"Download for table: {archive.resolve()}", flush=True)


if __name__ == "__main__":
    main()
