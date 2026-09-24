"""Evaluate five seed-55 BraTS models on small and medium ET lesions.

This reuses the project's 26-connected, one-to-one lesion matcher. The small
(10-44 voxels) and medium (45-4678 voxels) strata were fitted on the training
split and frozen before the 37-case test evaluation. Results are reported for
each stratum and for the pooled small+medium set. Unmatched GT lesions
contribute Dice=0 only to the GT-anchored mean. The historical nnUNet3d
surrogate is used for the fourth model, matching the original four-baseline
comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.wt_lesion_stratified import summarize_stratified_cases
from models.attunet3d import AttUNet3d
from models.nnunet3d import nnUNet3d
from models.resunet3d import ResUNet3d
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
from models.unet3d import UNet3d
from eval_wt_lesion_stratified import evaluation_dataloader, evaluate_model, find_checkpoint


BASE_KWARGS = {"in_channels": 4, "n_classes": 3, "n_channels": 24}
MODEL_SPECS = OrderedDict(
    (
        ("unet", ("3D U-Net", UNet3d, BASE_KWARGS, ("Unet", "UNet_model"), "baseline_unet3d")),
        ("resunet", ("3D ResUNet", ResUNet3d, BASE_KWARGS, ("ResUNet_model",), "baseline_resunet3d")),
        ("attention", ("3D Attention U-Net", AttUNet3d, BASE_KWARGS, (".autodl/attention-unet", "AttUNet_model"), "baseline_attention_unet3d")),
        ("nnunet", ("3D nnU-Net-style (nnUNet3d)", nnUNet3d, BASE_KWARGS, ("nnUnet", "nnUNet_model"), "baseline_nnunet3d")),
        ("afbms", ("AFBMS-ResUNet", ResUNetHFConcatBoundary, {**BASE_KWARGS, "multiscale_context_v2": True}, ("ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model",), "afbms_resunet")),
    )
)
STRATA = OrderedDict((
    ("small", (10, 44)),
    ("medium", (45, 4678)),
    ("large", (4679, None)),
))
TARGET_STRATA = ("small", "medium")
EXPECTED_TEST_LESIONS = {"small": 31, "medium": 36}
PACKAGED_WEIGHTS = (
    PROJECT_ROOT / "deliverables" / "AFBMS_ResUNet_handoff_20260907"
    / "artifacts" / "weights"
)


def parse_overrides(entries: list[str]) -> dict[str, Path]:
    overrides = {}
    for entry in entries:
        name, separator, path = entry.partition("=")
        if not separator or name not in MODEL_SPECS or not path:
            raise ValueError(
                f"invalid --checkpoint {entry!r}; use MODEL=PATH, where MODEL is "
                + ", ".join(MODEL_SPECS)
            )
        if name in overrides:
            raise ValueError(f"duplicate --checkpoint for {name}")
        overrides[name] = Path(path).expanduser()
    return overrides


def resolve_checkpoint(path: Path) -> Path | None:
    if path.is_file():
        if not path.name.startswith("best_model_") or path.suffix != ".pth":
            raise ValueError(f"expected best_model_*.pth, got {path}")
        return path
    if path.is_dir():
        found = find_checkpoint(str(path))
        return Path(found) if found else None
    return None


def checkpoints(autodl_root: Path, overrides: dict[str, Path]) -> dict[str, Path]:
    found = {}
    missing = []
    for name, (_, _, _, server_dirs, packaged_name) in MODEL_SPECS.items():
        candidates = (
            [overrides[name]] if name in overrides else
            [*(autodl_root / directory for directory in server_dirs),
             PACKAGED_WEIGHTS / packaged_name / "seed_55"]
        )
        checkpoint = next((value for path in candidates if (value := resolve_checkpoint(path))), None)
        if checkpoint is None:
            missing.append(f"{name}: " + " | ".join(str(path) for path in candidates))
        else:
            found[name] = checkpoint
    if missing:
        raise FileNotFoundError(
            "Missing best checkpoints; supply --checkpoint MODEL=PATH for each missing model:\n"
            + "\n".join(missing)
        )
    return found


def check_strata(path: Path) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if (metadata.get("region") != "ET"
            or metadata.get("fit_split") != "train"
            or metadata.get("connectivity") != 26
            or metadata.get("min_component_size") != 10
            or {name: tuple(metadata["strata"][name]) for name in STRATA} != dict(STRATA)):
        raise ValueError(f"strata do not match the formal ET test protocol: {path}")


def load_model(model_class, kwargs, checkpoint: Path, device):
    model = model_class(**kwargs)
    state = torch.load(checkpoint, map_location="cpu")
    for wrapper in ("state_dict", "model_state_dict"):
        if isinstance(state, dict) and wrapper in state:
            state = state[wrapper]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint has no state dict: {checkpoint}")
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.").replace("out.conv.0.", "out.conv.")
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return model.to(device).eval()


def pooled_summary(details, strata):
    """Pool per-GT-lesion rows across one or more frozen size strata."""
    selected = [row for row in details if row["stratum"] in strata]
    detected_rows = [row for row in selected if row["detected"]]
    gt_lesions = len(selected)
    detected = len(detected_rows)
    return {
        "gt_lesions": gt_lesions,
        "detected": detected,
        "missed": gt_lesions - detected,
        "lesion_recall": detected / gt_lesions if gt_lesions else float("nan"),
        "matched_lesion_dice": (
            sum(row["matched_dice"] for row in detected_rows) / detected
            if detected else float("nan")
        ),
        "gt_anchored_lesion_dice": (
            sum(row["gt_anchored_dice"] for row in selected) / gt_lesions
            if gt_lesions else float("nan")
        ),
    }


def result_row(label, model_id, checkpoint, test_cases, stratum, summary):
    ranges = {"small": "10-44", "medium": "45-4678", "small_medium": "10-4678"}
    return {
        "model": label,
        "model_id": model_id,
        "seed": 55,
        "split": "test",
        "stratum": stratum,
        "voxel_range": ranges[stratum],
        "checkpoint": str(checkpoint),
        "test_cases": test_cases,
        "gt_lesions": summary["gt_lesions"],
        "detected": summary["detected"],
        "missed": summary["missed"],
        "lesion_recall": summary["lesion_recall"],
        "matched_lesion_dice": summary["matched_lesion_dice"],
        "gt_anchored_lesion_dice": summary["gt_anchored_lesion_dice"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "tumourCSV.csv")
    parser.add_argument("--autodl-root", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--strata-json", type=Path, help="Optional frozen training ET strata to verify")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="MODEL=PATH",
                        help="Override a model's best checkpoint file or directory; repeat as needed")
    parser.add_argument("--output-dir", type=Path, default=Path("et_small_medium_lesion_five_models_results"))
    parser.add_argument("--device", default=None, help="cuda or cpu (default: choose automatically)")
    parser.add_argument("--dry-run", action="store_true", help="Check paths without loading MRI data")
    args = parser.parse_args()

    if args.strata_json:
        check_strata(args.strata_json)
    selected = checkpoints(args.autodl_root, parse_overrides(args.checkpoint))
    for name, path in selected.items():
        print(f"{name}: {path}")
    if args.dry_run:
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader, manifest = evaluation_dataloader(str(args.csv), "test")
    if len(manifest) != 37:
        raise ValueError(f"expected the original 37 test cases, got {len(manifest)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detail_rows = []
    reference_gt = {}
    for name, (label, model_class, kwargs, _, _) in MODEL_SPECS.items():
        checkpoint = selected[name]
        print(f"Evaluating {label} on {device}")
        model = load_model(model_class, kwargs, checkpoint, device)
        cases, details = evaluate_model(model, loader, device, 0.33, 10, 2, STRATA, "ET")
        summaries = summarize_stratified_cases(cases, STRATA)
        for stratum in TARGET_STRATA:
            summary = summaries[stratum]
            expected = EXPECTED_TEST_LESIONS[stratum]
            gt_identity = {
                (row["case_id"], row["gt_lesion_id"], row["gt_voxels"])
                for row in details if row["stratum"] == stratum
            }
            if summary["gt_lesions"] != expected or len(gt_identity) != expected:
                raise ValueError(
                    f"expected {expected} {stratum} ET GT lesions for {label}, "
                    f"got {summary['gt_lesions']}"
                )
            if stratum in reference_gt and gt_identity != reference_gt[stratum]:
                raise ValueError(f"GT {stratum}-lesion identities changed for {label}")
            reference_gt[stratum] = gt_identity
            summary_rows.append(result_row(
                label, name, checkpoint, len(manifest), stratum, summary
            ))
            print(
                f"  {stratum:<6} Matched={summary['matched_lesion_dice']:.4f}  "
                f"GT-anchored={summary['gt_anchored_lesion_dice']:.4f}  "
                f"Detected={summary['detected']}/{summary['gt_lesions']}"
            )

        combined = pooled_summary(details, TARGET_STRATA)
        summary_rows.append(result_row(
            label, name, checkpoint, len(manifest), "small_medium", combined
        ))
        print(
            f"  pooled Matched={combined['matched_lesion_dice']:.4f}  "
            f"GT-anchored={combined['gt_anchored_lesion_dice']:.4f}  "
            f"Detected={combined['detected']}/{combined['gt_lesions']}"
        )
        detail_rows.extend(
            {"model": label, "model_id": name, **row}
            for row in details if row["stratum"] in TARGET_STRATA
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)
    detail_frame = pd.DataFrame(detail_rows)
    detail_frame.to_csv(args.output_dir / "small_medium_lesions.csv", index=False)
    detail_frame[detail_frame["stratum"] == "small"].to_csv(
        args.output_dir / "small_lesions.csv", index=False
    )
    detail_frame[detail_frame["stratum"] == "medium"].to_csv(
        args.output_dir / "medium_lesions.csv", index=False
    )
    manifest.to_csv(args.output_dir / "test_cases.csv", index=False)
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
