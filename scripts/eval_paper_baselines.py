"""Evaluate pinned paper baselines on the project's frozen ET lesion strata.

Uses the exact matcher and 37-case test split used by
scripts/eval_et_small_lesion_five_models.py. Never evaluates the test set
during training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation.wt_lesion_stratified import summarize_stratified_cases
from eval_et_small_lesion_five_models import STRATA
from eval_wt_lesion_stratified import (
    evaluation_dataloader, evaluate_model, find_checkpoint, flatten_summaries,
)
from models.paper_baselines import PaperBaseline, SOURCES
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
from train_paper_baselines import sha256


class SlidingWindowModel(torch.nn.Module):
    def __init__(self, model, crop_size):
        super().__init__()
        self.model = model
        self.crop_size = crop_size

    def forward(self, images):
        return sliding_window_inference(
            images, tuple(self.crop_size), 1, self.model, overlap=0.25,
            mode="constant",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "tumourCSV.csv")
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/paper_baselines"))
    parser.add_argument("--source-dir", type=Path, default=Path("/root/autodl-tmp/paper_baseline_sources"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/paper_baselines/et_test_results"))
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--full-checkpoint", type=Path,
                        help="Optional FULL best_model_*.pth or its directory")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    args.csv = args.csv.expanduser().resolve()
    if args.device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to run full-volume evaluation on CPU")
        device = torch.device("cuda")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; check the NVIDIA driver and PyTorch CUDA build")
    loader, cases = evaluation_dataloader(str(args.csv), "test")
    actual_ids = cases["case_id"].astype(str).tolist()
    if len(actual_ids) != 37:
        raise RuntimeError(f"Expected 37 BraTS 2020 test cases, got {len(actual_ids)}")
    csv_digest = sha256(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, details, source_manifest = [], [], {}
    reference_small_gt = None

    def checked_summaries(case_results, detail_rows, label):
        nonlocal reference_small_gt
        summaries = summarize_stratified_cases(case_results, STRATA)
        small_gt = {(row["case_id"], row["gt_lesion_id"], row["gt_voxels"])
                    for row in detail_rows if row["stratum"] == "small"}
        if summaries["small"]["gt_lesions"] != 31 or len(small_gt) != 31:
            raise RuntimeError(f"{label}: expected 31 frozen small ET GT lesions")
        if reference_small_gt is not None and small_gt != reference_small_gt:
            raise RuntimeError(f"{label}: test GT lesions differ from the other models")
        reference_small_gt = small_gt
        return summaries

    for name in SOURCES:
        run_dir = args.run_root / name / f"seed_{args.seed}"
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        if (protocol["model"] != name or protocol["csv_sha256"] != csv_digest
                or protocol["seed"] != args.seed
                or protocol["split"]["test"] != actual_ids
                or protocol["upstream_commit"] != SOURCES[name][1]):
            raise RuntimeError(f"{name}: checkpoint training protocol does not match test data")
        checkpoint = find_checkpoint(str(run_dir))
        if checkpoint is None:
            raise FileNotFoundError(f"No best_model_*.pth in {run_dir}")
        print(f"Evaluating {name}: {checkpoint}", flush=True)
        model = PaperBaseline(name, args.source_dir.expanduser()).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        model.eval()
        if protocol["crop_size"] is not None:
            model = SlidingWindowModel(model, protocol["crop_size"])
        source_manifest[name] = {"checkpoint": checkpoint, "protocol": str(run_dir / "protocol.json")}
        # SuperLightNet samples a random viewing direction even in eval mode.
        # Fix its CPU/CUDA RNG immediately before the ordered test-case pass.
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        case_results, detail_rows = evaluate_model(model, loader, device, 0.33, 10, 2, STRATA, "ET")
        summaries = checked_summaries(case_results, detail_rows, name)
        rows.extend(flatten_summaries(name, summaries))
        details.extend({"model": name, **row} for row in detail_rows)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.full_checkpoint is not None:
        checkpoint = args.full_checkpoint.expanduser()
        if checkpoint.is_dir():
            found = find_checkpoint(str(checkpoint))
            if found is None:
                raise FileNotFoundError(f"No best_model_*.pth in {checkpoint}")
            checkpoint = Path(found)
        if not checkpoint.is_file() or not checkpoint.name.startswith("best_model_"):
            raise FileNotFoundError(f"Expected FULL best_model_*.pth: {checkpoint}")
        model = ResUNetHFConcatBoundary(4, 3, 24, multiscale_context_v2=True).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("module.").replace("out.conv.0.", "out.conv."): value
                 for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        source_manifest["FULL"] = {"checkpoint": str(checkpoint)}
        case_results, detail_rows = evaluate_model(model, loader, device, 0.33, 10, 2, STRATA, "ET")
        summaries = checked_summaries(case_results, detail_rows, "FULL")
        rows.extend(flatten_summaries("FULL", summaries))
        details.extend({"model": "FULL", **row} for row in detail_rows)

    pd.DataFrame(rows).to_csv(args.output_dir / "et_test_lesion_summary.csv", index=False)
    pd.DataFrame(details).to_csv(args.output_dir / "et_test_lesion_details.csv", index=False)
    cases.to_csv(args.output_dir / "et_test_cases.csv", index=False)
    (args.output_dir / "sources.json").write_text(
        json.dumps(source_manifest, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows)[["model", "stratum", "gt_lesions", "lesion_recall",
                              "matched_lesion_dice", "gt_anchored_lesion_dice"]].to_string(index=False))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
