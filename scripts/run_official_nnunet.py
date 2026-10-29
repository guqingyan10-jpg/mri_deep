"""Official nnU-Net v2 3d_fullres single-model system on the frozen study split.

Calls the installed official planner, trainer and inference entry points.
No replacement network, custom optimizer, shortened training, or random
five-fold split is introduced. Native MRI input is the default preparation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED_VERSION = "2.6.2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment(experiment: dict, seed: int, workers: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"nnUNet_raw": experiment["raw_dir"],
                "nnUNet_preprocessed": experiment["preprocessed_dir"],
                "nnUNet_results": str(Path(experiment["results_dir"]) / f"seed{seed}"),
                "nnUNet_n_proc_DA": str(workers), "nnUNet_compile": "false",
                "PYTHONHASHSEED": str(seed), "NO_ALBUMENTATIONS_UPDATE": "1",
                "PYTHONIOENCODING":"utf-8", "PYTHONUNBUFFERED":"1"})
    return env


def official_entry(kind: str, seed: int, arguments: list[str]) -> None:
    """Seed a fresh process before entering the unmodified official CLI."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    entry = {
        "plan": ("nnunetv2.experiment_planning.plan_and_preprocess_entrypoints", "plan_and_preprocess_entry"),
        "train": ("nnunetv2.run.run_training", "run_training_entry"),
        "predict": ("nnunetv2.inference.predict_from_raw_data", "predict_entry_point"),
    }
    from importlib import import_module
    module, name = entry[kind]
    sys.argv = [f"nnUNetv2_{kind}", *arguments]
    getattr(import_module(module), name)()
    if kind == "train" and os.environ.get("NNUNET_STUDY_RESOURCE_REPORT"):
        write_json(Path(os.environ["NNUNET_STUDY_RESOURCE_REPORT"]), {
            "wall_seconds": time.perf_counter()-started,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            "note": "Memory peak is for this trainer process; wall time includes its final validation."})


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "train", "predict", "evaluate", "all"), default="all")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--gpu-memory-target", type=float, default=8.0)
    parser.add_argument("--resume", action="store_true", help="Explicitly continue an interrupted official training run")
    parser.add_argument("--disable-tta", action="store_true", help="Disable official mirror TTA; default retains official inference TTA")
    parser.add_argument("--allow-nonstandard-cohort", action="store_true",
                        help="Only for an explicitly separate/synthetic cohort; the study split is checked by default")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_protocol(path: Path, allow_nonstandard: bool = False) -> dict:
    from scripts.eval_official_nnunet import load_experiment
    protocol = load_experiment(path, allow_nonstandard_cohort=allow_nonstandard)
    protocol.pop("_case_map", None)
    expected_name = f"Dataset{protocol['dataset_id']:03d}_"
    if not protocol["dataset_name"].startswith(expected_name):
        raise ValueError("Dataset name and ID do not agree")
    dataset_dir = Path(protocol["dataset_dir"])
    expected = set(protocol["split"]["train"] + protocol["split"]["valid"])
    actual = {p.name[:-7] for p in (dataset_dir / "labelsTr").glob("*.nii.gz")}
    if actual != expected or actual.intersection(protocol["split"]["test"]):
        raise ValueError("labelsTr must contain exactly the frozen train+valid cases, never test labels")
    for subset, ids in (("imagesTr", expected), ("imagesTs", set(protocol["split"]["test"]))):
        actual_images = {p.name for p in (dataset_dir/subset).glob("*.nii.gz")}
        expected_images = {f"{case_id}_{channel:04d}.nii.gz" for case_id in ids for channel in range(4)}
        if actual_images != expected_images:
            raise ValueError(f"{subset} must contain exactly the four frozen modalities per assigned case")
    dataset_json = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    if dataset_json != protocol["dataset_json"]:
        raise ValueError("Exported dataset.json was changed after preparation")
    if dataset_json.get("overwrite_image_reader_writer") != "SimpleITKIO":
        raise ValueError("SimpleITKIO is required for the documented CZYX probability layout")
    split = [{"train": protocol["split"]["train"], "val": protocol["split"]["valid"]}]
    if json.loads((path.parent / "splits_final.json").read_text(encoding="utf-8")) != split:
        raise ValueError("Frozen official split differs from experiment.json")
    return protocol


def commands(args, experiment: dict) -> dict[str, list[str]]:
    prefix = [sys.executable, str(Path(__file__).resolve()), "_entry"]
    dataset = str(experiment["dataset_id"])
    common = ["-d", dataset, "-c", "3d_fullres"]
    output = args.experiment.resolve().parent / "official_pipeline" / f"seed{args.seed}"
    plan = [*prefix, "plan", str(args.seed), "-d", dataset,
            "--verify_dataset_integrity", "-c", "3d_fullres", "-npfp", str(args.num_processes),
            "-np", str(args.num_processes), "-gpu_memory_target", str(args.gpu_memory_target)]
    train = [*prefix, "train", str(args.seed), dataset, "3d_fullres", "0",
             "-tr", "nnUNetTrainer", "-p", "nnUNetPlans", "-device", args.device]
    if args.resume:
        train.append("--c")
    predict = [*prefix, "predict", str(args.seed), *common,
               "-i", str(Path(experiment["dataset_dir"]) / "imagesTs"),
               "-o", str(output / "predictions"), "-f", "0", "-tr", "nnUNetTrainer", "-p", "nnUNetPlans",
               "-chk", "checkpoint_final.pth", "--save_probabilities", "--continue_prediction",
               "-device", args.device, "-npp", str(args.num_processes), "-nps", str(args.num_processes)]
    if args.disable_tta:
        predict.append("--disable_tta")
    return {"plan": plan, "train": train, "predict": predict}


def run_logged(command: list[str], env: dict, log: Path) -> float:
    start = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\nCOMMAND " + json.dumps(command) + "\n")
        with subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                              bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            result = process.wait()
    elapsed = time.perf_counter() - start
    if result:
        raise subprocess.CalledProcessError(result, command)
    return elapsed


def install_split(experiment: dict, source: Path) -> Path:
    destination = Path(experiment["preprocessed_dir"]) / experiment["dataset_name"] / "splits_final.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = json.loads(source.read_text(encoding="utf-8"))
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != expected:
        raise ValueError("Existing official split differs; refusing to replace it")
    write_json(destination, expected)
    return destination


def preprocessing_snapshot(experiment: dict) -> dict:
    base=Path(experiment["preprocessed_dir"])/experiment["dataset_name"]
    metadata_paths=[base/name for name in ("nnUNetPlans.json","dataset.json","dataset_fingerprint.json")]
    for path in metadata_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing official preprocessing metadata: {path}; run --stage plan")
    plans=json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    directory=base/plans["configurations"]["3d_fullres"]["data_identifier"]
    cases=experiment["split"]["train"]+experiment["split"]["valid"]
    expected=set(cases)
    actual={p.name[:-5]for p in directory.glob("*.b2nd")if not p.name.endswith("_seg.b2nd")}
    if actual != expected:
        raise ValueError("Preprocessed case inventory differs from frozen train+validation cases; rerun --stage plan")
    for case in cases:
        for suffix in (".b2nd","_seg.b2nd",".pkl"):
            if not (directory/f"{case}{suffix}").is_file():
                raise FileNotFoundError(f"Incomplete preprocessing for {case}; rerun --stage plan")
    return {"metadata_sha256":{path.name:digest(path)for path in metadata_paths},
            "data_identifier":directory.name,"case_ids":cases}


def runtime_info() -> dict:
    import torch
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {name: metadata.version(name) for name in
                         ["nnunetv2", "dynamic-network-architectures", "numpy", "scipy", "nibabel",
                          "batchgenerators", "batchgeneratorsv2", "blosc2", "SimpleITK", "scikit-learn"]},
            "environment_packages": {item.metadata["Name"]:item.version for item in metadata.distributions()
                                     if item.metadata.get("Name")}}


def checkpoint_summary(model_dir: Path) -> dict:
    """Read the official final checkpoint, without changing its training state."""
    import torch
    path = model_dir / "checkpoint_final.pth"
    state = torch.load(path, map_location="cpu", weights_only=False)
    epoch = int(state["current_epoch"])
    planned = json.loads((model_dir.parent / "plans.json").read_text(encoding="utf-8"))
    configuration = planned["configurations"]["3d_fullres"]
    return {"completed_epochs": epoch, "updates_per_epoch": 250,
            "completed_optimizer_updates": epoch*250,
            "planned_batch_size": configuration["batch_size"],
            "planned_patch_size": configuration["patch_size"],
            "parameters": sum(value.numel() for key,value in state["network_weights"].items()
                              if not key.endswith(("running_mean","running_var","num_batches_tracked"))),
            "parameter_note": "State tensor count; normalization running statistics excluded",
            "final_checkpoint_sha256": digest(path),
            "best_checkpoint_sha256": digest(model_dir/"checkpoint_best.pth")}


def package_results(out: Path, experiment_path: Path, model_dir: Path) -> Path:
    """A small download for manuscript review; no model weights or MRI arrays."""
    destination = out/"results_for_paper.zip"
    paths = [(p,p.relative_to(out)) for p in out.rglob("*") if p.is_file()
             and "predictions" not in p.relative_to(out).parts
             and p.suffix.lower() in {".json",".csv",".log"}]
    paths += [(experiment_path,Path("data_protocol/experiment.json")),
              (experiment_path.parent/"splits_final.json",Path("data_protocol/splits_final.json"))]
    paths += [(model_dir.parent/name,Path("official_model")/name)
              for name in ("plans.json","dataset.json","dataset_fingerprint.json")]
    paths += [(model_dir/"progress.png",Path("official_model/progress.png"))]
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path,name in paths:
            if path.is_file():archive.write(path,name.as_posix())
    return destination


def main(argv=None) -> None:
    args = cli().parse_args(argv)
    if args.num_processes < 1 or args.gpu_memory_target <= 0:
        raise ValueError("Worker count and GPU memory target must be positive")
    args.experiment = args.experiment.resolve()
    experiment = load_protocol(args.experiment,args.allow_nonstandard_cohort)
    tasks = commands(args, experiment)
    env = environment(experiment, args.seed, args.num_processes)
    out = args.experiment.parent / "official_pipeline" / f"seed{args.seed}"
    model_dir = Path(env["nnUNet_results"]) / experiment["dataset_name"] / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_0"
    env["NNUNET_STUDY_RESOURCE_REPORT"] = str(out/"training_resources.json")
    protocol = {"experiment_sha256": digest(args.experiment), "seed": args.seed,
                "dataset": experiment["dataset"], "model": "official nnU-Net v2 complete pipeline",
                "nnunetv2": EXPECTED_VERSION, "configuration": "3d_fullres", "trainer": "nnUNetTrainer",
                "fold": 0, "ensemble": False, "mirror_tta": not args.disable_tta,
                "checkpoint": "checkpoint_final.pth", "initialization": "random, no AFBMS/ResUNet warm-start",
                "training_input_space": experiment.get("input_space","study"),
                "allow_nonstandard_cohort": args.allow_nonstandard_cohort,
                "compile": False, "default_epoch_ceiling": 1000, "default_updates_per_epoch": 250,
                "seed_note": "Initial process seeds are fixed; GPU and multiworker training are not promised bitwise deterministic",
                "budget_note": "Official recipe is a system comparison, not an equal-update architecture ablation",
                "model_dir": str(model_dir)}
    print(json.dumps({"protocol": protocol, "environment": {k: env[k] for k in
                    ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "nnUNet_n_proc_DA"]},
                    "commands": tasks, "stage": args.stage, "dry_run": args.dry_run}, indent=2), flush=True)
    if args.dry_run:
        return
    if metadata.version("nnunetv2") != EXPECTED_VERSION:
        raise RuntimeError(f"Expected nnunetv2=={EXPECTED_VERSION}; install requirements_nnunet_official.txt in an isolated environment")
    import torch
    if not hasattr(torch,"GradScaler"):
        raise RuntimeError("nnunetv2==2.6.2 requires the modern torch.GradScaler API. "
                           "Install the documented PyTorch 2.5.1 environment; the original training environment can stay unchanged.")
    if args.device == "cuda" and args.stage in {"train","predict","all"}:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    protocol_path = out / "run_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8"))["protocol"] != protocol:
        raise ValueError("This output directory belongs to a different experiment/inference protocol")
    out.mkdir(parents=True, exist_ok=True)
    write_json(protocol_path, {"protocol": protocol, "runtime": runtime_info()})
    stages = ("plan", "train", "predict", "evaluate") if args.stage == "all" else (args.stage,)
    timing_path = out / "stage_history.json"
    history = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else []
    for stage in stages:
        start = time.perf_counter()
        entry = {"stage": stage, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            if stage == "plan":
                marker = args.experiment.parent / "official_planning.json"
                plan_protocol = {"experiment_sha256": digest(args.experiment), "gpu_memory_target": args.gpu_memory_target,
                                 "nnunetv2": EXPECTED_VERSION, "configuration": "3d_fullres"}
                if marker.exists():
                    if json.loads(marker.read_text(encoding="utf-8")) != plan_protocol:
                        raise ValueError("Existing planning protocol differs; choose a new work directory")
                    plans = Path(experiment["preprocessed_dir"]) / experiment["dataset_name"] / "nnUNetPlans.json"
                    if not plans.is_file():
                        raise FileNotFoundError("Planning marker exists but plans are missing")
                    print("Reusing completed official planning/preprocessing")
                else:
                    run_logged(tasks[stage], env, out / "plan.log")
                    plans = Path(experiment["preprocessed_dir"]) / experiment["dataset_name"] / "nnUNetPlans.json"
                    if not plans.is_file():
                        raise FileNotFoundError("Official planner returned without nnUNetPlans.json")
                    write_json(marker, plan_protocol)
                snapshot=preprocessing_snapshot(experiment)
                snapshot_path=args.experiment.parent/"official_planning_snapshot.json"
                if snapshot_path.exists() and json.loads(snapshot_path.read_text(encoding="utf-8"))!=snapshot:
                    raise ValueError("Completed official plans or dataset metadata were modified")
                write_json(snapshot_path,snapshot)
                install_split(experiment, args.experiment.parent / "splits_final.json")
            elif stage == "train":
                snapshot_path=args.experiment.parent/"official_planning_snapshot.json"
                if not snapshot_path.is_file():
                    raise FileNotFoundError("Missing preprocessing completion record; run --stage plan")
                if json.loads(snapshot_path.read_text(encoding="utf-8")) != preprocessing_snapshot(experiment):
                    raise ValueError("Preprocessing/plan identity changed after completion")
                install_split(experiment, args.experiment.parent / "splits_final.json")
                marker = out / "training_complete.json"
                if marker.exists() and (model_dir / "checkpoint_final.pth").is_file() and (model_dir / "checkpoint_best.pth").is_file():
                    saved = json.loads(marker.read_text(encoding="utf-8"))
                    if (saved.get("final_checkpoint_sha256") != digest(model_dir/"checkpoint_final.pth")
                            or saved.get("best_checkpoint_sha256") != digest(model_dir/"checkpoint_best.pth")):
                        raise ValueError("Completed official checkpoints were modified")
                    print("Reusing completed official training")
                else:
                    checkpoints = list(model_dir.glob("checkpoint_*.pth"))
                    if checkpoints and not args.resume:
                        raise ValueError("Existing training checkpoints found; use --resume explicitly")
                    if args.resume and not checkpoints:
                        raise ValueError("--resume requested without an official checkpoint")
                    run_logged(tasks[stage], env, out / "train.log")
                    if not (model_dir / "checkpoint_final.pth").is_file() or not (model_dir / "checkpoint_best.pth").is_file():
                        raise RuntimeError("Official training returned without final and best checkpoints")
                    budget = checkpoint_summary(model_dir)
                    if budget["completed_epochs"] != 1000:
                        raise RuntimeError("Official default training did not complete its 1000-epoch budget")
                    write_json(marker, {"status": "completed", "seed": args.seed,**budget})
            elif stage == "predict":
                if not (model_dir / "checkpoint_final.pth").is_file():
                    raise FileNotFoundError(model_dir / "checkpoint_final.pth")
                prediction_protocol = {**protocol, "checkpoint_sha256": digest(model_dir / "checkpoint_final.pth")}
                marker = out / "prediction_protocol.json"
                if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != prediction_protocol:
                    raise ValueError("Existing predictions came from a different checkpoint or protocol")
                write_json(marker, prediction_protocol)
                run_logged(tasks[stage], env, out / "predict.log")
            else:
                from scripts.eval_official_nnunet import evaluate_experiment
                prediction_marker = out/"prediction_protocol.json"
                if not prediction_marker.is_file():
                    raise ValueError("Missing prediction provenance; run the official predict stage first")
                recorded = json.loads(prediction_marker.read_text(encoding="utf-8"))
                if (recorded != {**protocol,"checkpoint_sha256":digest(model_dir/"checkpoint_final.pth")}):
                    raise ValueError("Prediction provenance no longer matches the model/checkpoint")
                for decision, name in [("probabilities", "evaluation_study033"), ("native-mask", "evaluation_native_default")]:
                    evaluate_experiment(args.experiment, out / "predictions", out / name,
                                        decision=decision, threshold=0.33, probability_layout="official",
                                        allow_nonstandard_cohort=args.allow_nonstandard_cohort,
                                        model_name="Official nnU-Net v2 complete pipeline")
            entry["status"] = "completed"
        except BaseException as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            entry["wall_seconds"] = time.perf_counter() - start
            history.append(entry)
            write_json(timing_path, history)
    if "evaluate" in stages:
        print(f"Download for manuscript review: {package_results(out,args.experiment,model_dir)}",flush=True)
    print(f"Official system baseline outputs: {out}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    if len(sys.argv) > 1 and sys.argv[1] == "_entry":
        official_entry(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
    else:
        main()
