"""Fill only missing entries of the seven-model BraTS2020 comparison table.

The five historical six-column results below are transcribed from the supplied
table, not recomputed.  Three missing matched small-lesion values are taken
from a verified prior summary when available, otherwise from the original
five-model checkpoints.  The two new paper models are evaluated once each on
the frozen 37-case test split; that pass computes all seven table columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TABLE = [
    ("unet", "3D U-Net", .7935, .7494, 16.24, .7415, .4315, .0687, None),
    ("resunet", "3D ResUNet", .8207, .7585, 10.26, .7503, .5213, .0357,
     .3684612447499045),
    ("attention", "3D Attention U-Net", .7743, .7317, 16.93, .7062, .5081,
     .0076, None),
    ("nnunet", "nnU-Net 风格 3D 网络", .7972, .7577, 11.11, .7640,
     .5143, .0336, None),
    ("afbms", "AFBMS-ResUNet", .8252, .7764, 8.29, .7871, .5385, .0235,
     .3647058823529412),
    ("doubleblock", "DoubleBlock-ViT", None, None, None, None, None, None, None),
    ("superlight", "SuperLightNet", None, None, None, None, None, None, None),
]
FIELDS = ("model_id", "method", "macro_dice", "et_dice", "et_hd95_mm",
          "boundary_dice_et_nsd_1mm", "lesion_f1_26conn_micro",
          "small_lesion_gt_anchored_dice", "small_lesion_matched_dice")
OLD_IDS = ("unet", "resunet", "attention", "nnunet", "afbms")
NEW_IDS = ("doubleblock", "superlight")
METRIC_VERSION = "seven-model-table-v1: threshold=.33; ET26/min10; small=10..44"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_template():
    return [dict(zip(FIELDS, values)) for values in TABLE]


def rounded_agrees(actual, displayed):
    """The historical table stores four decimal places, not raw values."""
    return math.isfinite(actual) and abs(actual - displayed) <= .000050001


def put_matched(rows, model_id, matched, anchored, source):
    row = next(item for item in rows if item["model_id"] == model_id)
    if not rounded_agrees(float(anchored), row["small_lesion_gt_anchored_dice"]):
        raise ValueError(
            f"{model_id}: cached GT-anchored Dice {anchored} disagrees with the "
            f"existing table {row['small_lesion_gt_anchored_dice']}; check cohort/checkpoint"
        )
    value = float(matched)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{model_id}: invalid matched Dice {matched}")
    row["small_lesion_matched_dice"] = value
    print(f"Reused {model_id} matched Dice from {source}: {value:.6f}", flush=True)


def format_value(value, key):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.2f}" if key == "et_hd95_mm" else f"{value:.4f}"


def pooled_lesion_f1(tp, fp, fn):
    """The displayed old F1 is pooled 26-connected F1, not per-case mean F1."""
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else float("nan")


def write_table(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "seven_model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    columns = (
        ("method", "方法"), ("macro_dice", "Macro Dice ↑"),
        ("et_dice", "ET Dice ↑"), ("et_hd95_mm", "ET HD95 (mm) ↓"),
        ("boundary_dice_et_nsd_1mm", "Boundary Dice (ET NSD, 1 mm) ↑"),
        ("lesion_f1_26conn_micro", "病灶 F1 ↑"),
        ("small_lesion_gt_anchored_dice", "小病灶 Dice (GT 锚定) ↑"),
        ("small_lesion_matched_dice", "小病灶 Dice (matched) ↑"),
    )
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(
            row[key] if key == "method" else format_value(row[key], key)
            for key, _ in columns
        ) + " |")
    lines.extend((
        "",
        "口径：BraTS2020 固定 37 例测试集，阈值 0.33。Boundary Dice 沿用旧表的 "
        "ET normalized surface Dice（NSD，容差 1 mm）；病灶 F1 为 ET 26 连通、"
        "最小 10 体素的一对一匹配后 pooled/micro F1。小病灶为训练集预先确定的 "
        "10–44 体素 GT ET 病灶；GT 锚定 Dice 对漏检计 0，matched Dice 仅平均"
        "成功匹配的 GT 病灶。— 表示尚未取得可验证结果。",
        "",
        "前五行原有六列逐字保留用户给出的四位小数，不代表重新评估；"
        "两个新增模型须以各自训练完成的 best_model_*.pth 计算。",
    ))
    (output_dir / "seven_model_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Table: {csv_path}", flush=True)


def read_legacy_summary(path, case_ids, rows):
    if not path.is_file():
        return set()
    manifest = path.with_name("test_cases.csv")
    if not manifest.is_file():
        raise FileNotFoundError(f"Existing summary needs its test_cases.csv: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        cached_ids = [str(row["case_id"]) for row in csv.DictReader(handle)]
    if cached_ids != case_ids:
        raise ValueError(f"Legacy summary test cases/order differ: {manifest}")
    reused = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for result in csv.DictReader(handle):
            name = result["model_id"]
            if name not in OLD_IDS or result["stratum"] != "small":
                continue
            if (int(result["seed"]) != 55 or result["split"] != "test"
                    or int(result["test_cases"]) != 37
                    or int(result["gt_lesions"]) != 31):
                raise ValueError(f"{name}: legacy summary protocol differs")
            if name in reused:
                raise ValueError(f"duplicate small row in {path}: {name}")
            put_matched(rows, name, result["matched_lesion_dice"],
                        result["gt_anchored_lesion_dice"], str(path))
            reused.add(name)
    return reused


def metric_cache(cache_path, checkpoint, csv_digest, case_ids, protocol_digest=None):
    if not cache_path.is_file():
        return None
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if (cached.get("version") != METRIC_VERSION
            or cached.get("csv_sha256") != csv_digest
            or cached.get("test_ids") != case_ids
            or cached.get("protocol_sha256") != protocol_digest
            or cached.get("checkpoint_sha256") != sha256(checkpoint)):
        return None
    return cached["metrics"]


def save_metric_cache(path, checkpoint, csv_digest, case_ids, metrics,
                      protocol_digest=None):
    payload = {"version": METRIC_VERSION, "csv_sha256": csv_digest,
               "test_ids": case_ids, "checkpoint": str(checkpoint),
               "protocol_sha256": protocol_digest,
               "checkpoint_sha256": sha256(checkpoint), "metrics": metrics}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def require_finished_training(run_dir, protocol):
    """Best weights exist mid-run; a test score needs a finished run."""
    log_path = run_dir / "train_log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(f"training log missing: {log_path}")
    with log_path.open(newline="", encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))
    if not entries:
        raise RuntimeError(f"empty training log: {log_path}")
    last = entries[-1]
    epoch, bad = int(last["epoch"]), int(last["bad_epochs"])
    if epoch < int(protocol["epochs"]) and bad < int(protocol["early_stopping_patience"]):
        raise RuntimeError(
            f"{run_dir}: training not complete (epoch {epoch}, bad epochs {bad}); "
            "wait for max epochs or early stopping before testing"
        )


def evaluate_new(model, loader, case_ids, device, seed):
    """One forward pass per test case for all seven table columns."""
    import numpy as np
    import torch
    from scipy import ndimage
    from evaluation.advanced_metrics import hd95_single, nsd_single
    from evaluation.wt_lesion_stratified import (
        match_lesion_components, summarize_stratified_cases,
    )
    from eval_et_small_lesion_five_models import STRATA

    torch.manual_seed(seed)  # SuperLightNet samples a view during inference.
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    dice = {name: [] for name in ("WT", "TC", "ET")}
    hd95, nsd, lesions = [], [], []
    structure = ndimage.generate_binary_structure(3, 3)
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            case_id = str(batch["Id"][0])
            if case_id != case_ids[index]:
                raise RuntimeError(f"test order differs at {index}: {case_id}")
            images = batch["image"].to(device)
            raw = model(images)
            if isinstance(raw, tuple):
                raw = raw[0]
            predicted = (torch.sigmoid(raw) >= .33).cpu().numpy()[0]
            target = batch["mask"].numpy()[0]
            for channel, name in enumerate(("WT", "TC", "ET")):
                pred, gt = predicted[channel], target[channel]
                union = float(pred.sum() + gt.sum())
                overlap = float((pred * gt).sum())
                dice[name].append(2 * overlap / (union + 1e-9) if union else 1.0)
            hd95.append(hd95_single(predicted[2], target[2]))
            nsd.append(nsd_single(predicted[2], target[2], tau=1.0))
            result = match_lesion_components(
                predicted[2], target[2], structure=structure, min_component_size=10)
            result["case_id"] = case_id
            lesions.append(result)
            print(f"test {index+1}/{len(case_ids)} {case_id}", flush=True)
    summaries = summarize_stratified_cases(lesions, STRATA)
    if (summaries["small"]["gt_lesions"] != 31
            or summaries["medium"]["gt_lesions"] != 36
            or summaries["large"]["gt_lesions"] != 26):
        raise RuntimeError("frozen ET GT lesion counts differ from the five-model test")
    all_lesions = summaries["all"]
    tp, fp, fn = (all_lesions["detected"], all_lesions["fp"],
                  all_lesions["missed"])
    finite_hd = [float(value) for value in hd95 if math.isfinite(value)]
    finite_nsd = [float(value) for value in nsd if math.isfinite(value)]
    if not finite_hd or not finite_nsd:
        raise RuntimeError("no defined ET HD95 or NSD values")
    class_mean = {name: float(np.mean(values)) for name, values in dice.items()}
    matched = summaries["small"]["matched_lesion_dice"]
    if not math.isfinite(matched):
        matched = "N/A (0 matched)"
    return {
        "macro_dice": float(np.mean(list(class_mean.values()))),
        "et_dice": class_mean["ET"],
        "et_hd95_mm": float(np.mean(finite_hd)),
        "boundary_dice_et_nsd_1mm": float(np.mean(finite_nsd)),
        "lesion_f1_26conn_micro": pooled_lesion_f1(tp, fp, fn),
        "small_lesion_gt_anchored_dice": summaries["small"]["gt_anchored_lesion_dice"],
        "small_lesion_matched_dice": matched,
        "small_gt_lesions": 31, "small_detected": summaries["small"]["detected"],
        "et_hd95_defined_cases": len(finite_hd),
        "et_nsd_defined_cases": len(finite_nsd),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "tumourCSV.csv")
    parser.add_argument("--run-root", type=Path,
                        default=Path("/root/autodl-tmp/paper_baselines"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("/root/autodl-tmp/paper_baseline_sources"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/root/autodl-tmp/paper_baselines/comparison_table"))
    parser.add_argument("--legacy-summary", type=Path,
                        default=ROOT / "et_small_medium_lesion_five_models_results" / "summary.csv")
    parser.add_argument("--checkpoint", action="append", default=[],
                        metavar="OLD_MODEL=PATH", help="Override an old model checkpoint")
    parser.add_argument("--model", action="append", choices=NEW_IDS,
                        help="Evaluate only this new model; repeat for both (default)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prepare", action="store_true",
                        help="Write the pending table without loading MRI or PyTorch")
    parser.add_argument("--only-new", action="store_true",
                        help="Score ready paper models now; leave missing old matched cells blank")
    args = parser.parse_args()
    for key in ("csv", "run_root", "source_dir", "output_dir", "legacy_summary"):
        setattr(args, key, getattr(args, key).expanduser())
    rows = rows_template()
    write_table(rows, args.output_dir)
    if args.prepare:
        return

    import gc
    import torch
    from eval_et_small_lesion_five_models import (
        MODEL_SPECS, PACKAGED_WEIGHTS, STRATA, load_model, parse_overrides,
        resolve_checkpoint,
    )
    from eval_wt_lesion_stratified import evaluation_dataloader, evaluate_model
    from evaluation.wt_lesion_stratified import summarize_stratified_cases
    from models.paper_baselines import PaperBaseline, SOURCES
    from eval_paper_baselines import SlidingWindowModel

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device.type != "cuda":
        raise RuntimeError("full-volume test inference requires CUDA")
    loader, manifest = evaluation_dataloader(str(args.csv), "test")
    case_ids = manifest["case_id"].astype(str).tolist()
    if len(case_ids) != 37 or len(set(case_ids)) != 37:
        raise RuntimeError("expected the original 37 unique BraTS test cases")
    csv_digest = sha256(args.csv)

    # The historical table's six existing entries per model are never inferred.
    # Reuse the five-model summary if it exists. Two published matched values
    # are already in TABLE; only the three still missing require old inference.
    if not args.only_new:
        read_legacy_summary(args.legacy_summary, case_ids, rows)
    overrides = parse_overrides(args.checkpoint)
    for name in (() if args.only_new else OLD_IDS):
        row = next(item for item in rows if item["model_id"] == name)
        if row["small_lesion_matched_dice"] is not None:
            continue
        label, model_class, kwargs, server_dirs, packaged_name = MODEL_SPECS[name]
        candidates = ([overrides[name]] if name in overrides else
                      [*(Path("/root/autodl-tmp") / directory for directory in server_dirs),
                       PACKAGED_WEIGHTS / packaged_name / "seed_55"])
        checkpoint = next((found for path in candidates
                           if (found := resolve_checkpoint(path)) is not None), None)
        if checkpoint is None:
            raise FileNotFoundError(
                f"{name} matched Dice missing; supply --legacy-summary or "
                f"--checkpoint {name}=/path/to/best_model_*.pth")
        cache_path = args.output_dir / f"{name}_matched.json"
        cached = metric_cache(cache_path, checkpoint, csv_digest, case_ids)
        if cached is None:
            print(f"Only evaluating missing matched Dice for {label}", flush=True)
            model = load_model(model_class, kwargs, checkpoint, device)
            case_results, _ = evaluate_model(model, loader, device, .33, 10, 2, STRATA, "ET")
            summary = summarize_stratified_cases(case_results, STRATA)["small"]
            if summary["gt_lesions"] != 31:
                raise RuntimeError(f"{name}: expected 31 small GT lesions")
            cached = {"small_lesion_matched_dice": summary["matched_lesion_dice"],
                      "small_lesion_gt_anchored_dice": summary["gt_anchored_lesion_dice"]}
            save_metric_cache(cache_path, checkpoint, csv_digest, case_ids, cached)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        put_matched(rows, name, cached["small_lesion_matched_dice"],
                    cached["small_lesion_gt_anchored_dice"], str(cache_path))
        write_table(rows, args.output_dir)

    selected = tuple(dict.fromkeys(args.model or NEW_IDS))
    for name in selected:
        run_dir = args.run_root / name / "seed_55"
        protocol_path = run_dir / "protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(f"{name}: training not finished / protocol missing: {protocol_path}")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (protocol["model"] != name or protocol["seed"] != 55
                or protocol["csv_sha256"] != csv_digest
                or protocol["split"]["test"] != case_ids
                or protocol["upstream_commit"] != SOURCES[name][1]):
            raise RuntimeError(f"{name}: training protocol/test data mismatch")
        require_finished_training(run_dir, protocol)
        from eval_wt_lesion_stratified import find_checkpoint
        checkpoint_name = find_checkpoint(str(run_dir))
        if checkpoint_name is None:
            raise FileNotFoundError(f"{name}: no best_model_*.pth in {run_dir}")
        checkpoint = Path(checkpoint_name)
        cache_path = args.output_dir / f"{name}_metrics.json"
        protocol_digest = sha256(protocol_path)
        metrics = metric_cache(cache_path, checkpoint, csv_digest, case_ids,
                               protocol_digest)
        if metrics is None:
            print(f"Evaluating all missing table columns for {name}: {checkpoint}", flush=True)
            model = PaperBaseline(name, args.source_dir).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device,
                                             weights_only=True), strict=True)
            model.eval()
            if protocol["crop_size"] is not None:
                model = SlidingWindowModel(model, protocol["crop_size"])
            metrics = evaluate_new(model, loader, case_ids, device, 55)
            save_metric_cache(cache_path, checkpoint, csv_digest, case_ids,
                              metrics, protocol_digest)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"Reusing {name} metrics: {cache_path}", flush=True)
        row = next(item for item in rows if item["model_id"] == name)
        row.update({key: metrics[key] for key in FIELDS[2:]})
        write_table(rows, args.output_dir)
    print("Completed table; inspect seven_model_comparison.md and .csv")


if __name__ == "__main__":
    main()
