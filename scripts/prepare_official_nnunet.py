"""Prepare native MRI for official nnU-Net, retaining the study's frozen split.

Native inputs retain their original resolution and intensity values. The
official pipeline performs its own normalization, planning and resampling.
Separate study-grid GT is exported for common evaluation; it is never used
as native training input. --input-space study is an explicitly labelled
alternative, not the default official native-input system comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from data.dataset import BratsDataset, get_dataloader
from data.ucsf_bmsr import assign_grouped_splits, patient_group, read_manifest


BRATS_MODALITIES = ("flair", "t1", "t1ce", "t2")
UCSF_MODALITIES = ("T1post", "T1pre", "FLAIR", "subtraction")
BRATS_CROP_START_XYZ = (40, 40, 20)
STUDY_SHAPE_DHW = (100, 170, 170)
EXPECTED_COUNTS = {
    "brats": {"train": 257, "valid": 74, "test": 37},
    "ucsf": {"train": 321, "valid": 94, "test": 46},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_case_id(value: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Unsafe or unsupported case ID: {value!r}")
    return value


def source_signature(paths: list[Path]) -> list[dict]:
    signature = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")
        stat = path.stat()
        signature.append({"path": str(path.resolve()), "bytes": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns})
    return signature


def resolve_nifti(directory: Path, stem: str) -> Path:
    candidates = [directory / (stem + suffix) for suffix in (".nii", ".nii.gz")]
    available = [path for path in candidates if path.is_file()]
    if not available:
        raise FileNotFoundError(f"Missing NIfTI {stem} (.nii or .nii.gz) in {directory}")
    # Match the original loader when an uncompressed and compressed copy coexist.
    return available[0].resolve()


def brats_rows(csv_path: Path, data_root: Path | None) -> list[dict]:
    frame = pd.read_csv(csv_path)
    if not {"Brats20ID", "path"}.issubset(frame.columns):
        raise ValueError("BraTS CSV must contain Brats20ID and path columns")
    if frame["Brats20ID"].isna().any() or frame["Brats20ID"].duplicated().any():
        raise ValueError("BraTS CSV contains missing or duplicate case IDs")
    # Reuse the existing loader's split, including its CSV order and split seed.
    assignments = {}
    for split in ("train", "valid", "test"):
        loader = get_dataloader(BratsDataset, str(csv_path), split, batch_size=1, num_workers=0)
        assignments.update({str(value): (split,index) for index,value in enumerate(loader.dataset.df["Brats20ID"])})
    rows = []
    for record in frame.to_dict("records"):
        case_id = safe_case_id(record["Brats20ID"])
        directory = data_root / case_id if data_root is not None else Path(record["path"])
        images = [resolve_nifti(directory, f"{case_id}_{name}") for name in BRATS_MODALITIES]
        label = resolve_nifti(directory, f"{case_id}_seg")
        rows.append({"case_id": case_id, "split": assignments[case_id][0], "split_index": assignments[case_id][1],
                     "patient_group": case_id, "source_files": [*images, label]})
    return rows


def ucsf_rows(manifest_path: Path, prepared_root: Path | None,
              raw_root: Path | None = None, input_space: str = "study") -> list[dict]:
    frame = read_manifest(manifest_path)
    rows = []
    for record in frame.to_dict("records"):
        case_id = safe_case_id(record["subject_id"])
        group = patient_group(case_id)
        if str(record["patient_group"]) != group:
            raise ValueError(f"Patient group does not match subject ID for {case_id}")
        if prepared_root is None:
            path = Path(record["prepared_path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
        else:
            path = prepared_root / f"{case_id}.npz"
            if not path.is_file():
                path = prepared_root / "cases" / f"{case_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared UCSF case is missing: {path}")
        files = [path.resolve()]
        if input_space == "native":
            if raw_root is not None:
                directory = raw_root / case_id
            elif isinstance(record.get("source_dir"), str) and record["source_dir"]:
                directory = Path(record["source_dir"])
                if not directory.is_absolute():
                    directory = manifest_path.parent / directory
            else:
                raise ValueError("Native UCSF export requires source_dir in the manifest or --ucsf-data-root")
            files = [resolve_nifti(directory, f"{case_id}_{name}") for name in UCSF_MODALITIES]
            files += [resolve_nifti(directory, f"{case_id}_seg"), path.resolve()]
        rows.append({"case_id": case_id, "split": str(record["split"]),
                     "patient_group": group, "source_files": files,
                     "study_cache_file": path.resolve()})
    return rows


def validate_cohort(rows: list[dict], dataset: str, allow_nonstandard: bool) -> dict:
    if not rows:
        raise ValueError("Empty cohort")
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    split = {name: [row["case_id"] for row in rows if row["split"] == name]
             for name in ("train", "valid", "test")}
    if all("split_index"in row for row in rows):
        split = {name:[row["case_id"]for row in sorted(
            (row for row in rows if row["split"]==name),key=lambda row:row["split_index"])]for name in split}
    if any(not values for values in split.values()):
        raise ValueError("Every frozen train/valid/test split must contain cases")
    if set(row["split"] for row in rows) != set(split):
        raise ValueError("Unknown split value")
    groups = {}
    for row in rows:
        groups.setdefault(row["patient_group"], set()).add(row["split"])
    if any(len(values) != 1 for values in groups.values()):
        raise ValueError("Patient leakage across frozen splits")
    counts = {name: len(values) for name, values in split.items()}
    if not allow_nonstandard and counts != EXPECTED_COUNTS[dataset]:
        raise ValueError(f"Study cohort mismatch: {counts}; expected {EXPECTED_COUNTS[dataset]}. "
                         "Use --allow-nonstandard-cohort only for an explicitly separate experiment.")
    if dataset == "ucsf" and not allow_nonstandard:
        patient_counts = Counter(next(iter(values)) for values in groups.values())
        if dict(patient_counts) != {"train": 219, "valid": 63, "test": 32}:
            raise ValueError(f"UCSF frozen patient counts differ from the study: {patient_counts}")
        expected_assignment = assign_grouped_splits(ids,seed=10)
        if any(row["split"] != expected_assignment[row["case_id"]] for row in rows):
            raise ValueError("UCSF manifest differs from the study's original seed-10 patient split")
    return split


def check_image_geometry(image: nib.Nifti1Image, reference: nib.Nifti1Image, path: Path) -> None:
    if len(image.shape) != 3 or tuple(image.shape) != tuple(reference.shape):
        raise ValueError(f"Image/label shape mismatch for {path}: {image.shape} vs {reference.shape}")
    if not np.allclose(image.affine, reference.affine, rtol=1e-5, atol=1e-3):
        raise ValueError(f"Image/label affine mismatch: {path}")


def brats_arrays(row: dict, input_space: str = "study") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = nib.load(str(row["source_files"][0]))
    if len(reference.shape) != 3:
        raise ValueError(f"Expected a 3D BraTS image: {row['case_id']}")
    if any(size < end for size, end in zip(reference.shape, (210, 210, 120))):
        raise ValueError(f"BraTS image does not cover the study crop: {reference.shape}")
    channels = []
    for path in row["source_files"][:4]:
        image = nib.load(str(path))
        check_image_geometry(image, reference, path)
        source = np.asarray(image.dataobj)
        if not np.isfinite(source).all():
            raise ValueError(f"Non-finite source MRI values: {path}")
        if input_space == "native":
            channels.append(np.asarray(source, dtype=np.float32))
        else:
            cropped = BratsDataset.resize(None, source)
            if np.max(cropped) <= np.min(cropped):
                raise ValueError(f"Constant BraTS channel would produce non-finite study normalization: {path}")
            normalized = BratsDataset.normalize(None, cropped)
            channels.append(np.asarray(normalized, dtype=np.float32))
    label_path = row["source_files"][4]
    source_label = nib.load(str(label_path))
    check_image_geometry(source_label, reference, label_path)
    label = np.asarray(source_label.dataobj)
    if not np.isfinite(label).all() or not set(np.unique(label)).issubset({0, 1, 2, 4}):
        raise ValueError(f"Expected original BraTS labels 0/1/2/4: {label_path}")
    label = (label if input_space == "native" else BratsDataset.resize(None, label)).astype(np.uint8)
    label[label == 4] = 3
    translation = np.eye(4)
    translation[:3, 3] = BRATS_CROP_START_XYZ if input_space == "study" else (0, 0, 0)
    affine = reference.affine @ translation
    return np.stack(channels), label, affine


def ucsf_arrays(row: dict, allow_nonstandard: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(row.get("study_cache_file", row["source_files"][-1]), allow_pickle=False) as prepared:
        channels = np.asarray(prepared["image"], dtype=np.float32)
        mask = np.asarray(prepared["mask"])
        spacing = np.asarray(prepared["spacing_dhw"], dtype=np.float64)
    if channels.ndim != 4 or channels.shape[0] != 4 or mask.shape != (1, *channels.shape[1:]):
        raise ValueError(f"UCSF cache image/mask shapes are invalid for {row['case_id']}")
    if not allow_nonstandard and tuple(channels.shape[1:]) != STUDY_SHAPE_DHW:
        raise ValueError(f"Unexpected UCSF prepared space for {row['case_id']}: {channels.shape}")
    if not np.isfinite(channels).all() or not np.isfinite(mask).all():
        raise ValueError(f"Non-finite UCSF cache values for {row['case_id']}")
    if not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError(f"Expected a binary UCSF cache mask: {row['case_id']}")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError(f"Invalid effective spacing for {row['case_id']}: {spacing}")
    # The study cache is already canonical RAS and resized. Its common prepared
    # grid has no original-world origin; choose origin 0 and preserve its exact
    # effective voxel spacing. Images and labels share this declared geometry.
    affine = np.diag([*spacing[::-1], 1.0])
    return channels.transpose(0, 3, 2, 1), mask[0].transpose(2, 1, 0).astype(np.uint8), affine


def ucsf_native_arrays(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse canonical orientation, without study resizing or normalization."""
    reference = nib.as_closest_canonical(nib.load(str(row["source_files"][0])))
    channels = []
    for path in row["source_files"][:4]:
        image = nib.as_closest_canonical(nib.load(str(path)))
        check_image_geometry(image, reference, path)
        array = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite native MRI values: {path}")
        channels.append(array)
    label_path = row["source_files"][4]
    image = nib.as_closest_canonical(nib.load(str(label_path)))
    check_image_geometry(image, reference, label_path)
    label = np.asarray(image.dataobj)
    if not np.isfinite(label).all() or not set(np.unique(label)).issubset({0, 1}):
        raise ValueError(f"Expected binary native UCSF label: {label_path}")
    return np.stack(channels), label.astype(np.uint8), reference.affine


def dataset_description(dataset: str, training_cases: int) -> dict:
    if dataset == "brats":
        channels = BRATS_MODALITIES
        labels = {"background": 0, "WT": [1, 2, 3], "TC": [1, 3], "ET": 3}
        order = [2, 1, 3]
    else:
        channels = UCSF_MODALITIES
        labels = {"background": 0, "BM": 1}
        order = None
    description = {"channel_names": {str(i): name for i, name in enumerate(channels)},
            "labels": labels,
            "numTraining": training_cases, "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO"}
    if order is not None:
        description["regions_class_order"] = order
    return description


def export_case(row: dict, dataset: str, dataset_dir: Path, allow_nonstandard: bool,
                input_space: str = "study") -> dict:
    images, label, affine = (brats_arrays(row, input_space) if dataset == "brats" else
                            ucsf_native_arrays(row) if input_space == "native" else
                            ucsf_arrays(row, allow_nonstandard))
    is_test = row["split"] == "test"
    image_dir = dataset_dir / ("imagesTs" if is_test else "imagesTr")
    label_dir = dataset_dir / ("labelsTs_reference" if is_test else "labelsTr")
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [image_dir / f"{row['case_id']}_{index:04d}.nii.gz" for index in range(4)]
    label_path = label_dir / f"{row['case_id']}.nii.gz"
    for array, path in zip(images, image_paths):
        nib.save(nib.Nifti1Image(array.astype(np.float32), affine), str(path))
    nib.save(nib.Nifti1Image(label, affine), str(label_path))
    native_shape, native_affine = label.shape, affine.copy()
    native_label_file = label_path
    alignment = "identity"
    if input_space == "native":
        if dataset == "brats":
            label = BratsDataset.resize(None, label)
            translation = np.eye(4)
            translation[:3, 3] = BRATS_CROP_START_XYZ
            affine = affine @ translation
            alignment = "brats-study-crop"
        else:
            _, study_label, study_affine = ucsf_arrays(row, allow_nonstandard)
            # The original cache defines the scoring grid. Check that it really
            # comes from this native label and the existing nearest resize.
            from scripts.prepare_ucsf_bmsr import resize_dhw
            restored = resize_dhw(label.transpose(2, 1, 0).astype(np.float32),
                                  study_label.shape[::-1], "nearest")
            if not np.array_equal(restored.transpose(2, 1, 0)>0.5, study_label>0):
                raise ValueError(f"Native UCSF GT does not reproduce the existing study cache: {row['case_id']}")
            expected_spacing = nib.affines.voxel_sizes(native_affine)*np.asarray(native_shape)/np.asarray(study_label.shape)
            if not np.allclose(expected_spacing, nib.affines.voxel_sizes(study_affine), rtol=1e-4, atol=1e-4):
                raise ValueError(f"Native/cache UCSF effective spacing differs: {row['case_id']}")
            label, affine = study_label, study_affine
            alignment = "ucsf-canonical-study-resize"
        evaluation_dir = dataset_dir.parents[1] / "evaluation_targets"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        label_path = evaluation_dir / f"{row['case_id']}.nii.gz"
        nib.save(nib.Nifti1Image(label.astype(np.uint8), affine), str(label_path))
    spacing = nib.affines.voxel_sizes(affine)[::-1]
    return {"case_id": row["case_id"], "split": row["split"],
            "patient_group": row["patient_group"],
            "image_files": [str(path.resolve()) for path in image_paths],
            "label_file": str(label_path.resolve()),
            "native_label_file": str(native_label_file.resolve()),
            "prediction_reference_file": str(image_paths[0].resolve()),
            "prediction_alignment": alignment,
            "native_shape_dhw": list(native_shape[::-1]),
            "native_affine": native_affine.tolist(),
            "spacing_dhw": spacing.tolist(), "shape_dhw": list(label.shape[::-1])}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True, choices=("brats", "ucsf"))
    result.add_argument("--work-dir", required=True, type=Path)
    result.add_argument("--dataset-id", type=int)
    result.add_argument("--brats-csv", type=Path, default=REPO_ROOT / "tumourCSV.csv")
    result.add_argument("--brats-data-root", type=Path)
    result.add_argument("--ucsf-manifest", type=Path)
    result.add_argument("--ucsf-prepared-root", type=Path)
    result.add_argument("--ucsf-data-root", type=Path,
                        help="Native UCSF root containing subject directories; overrides manifest source_dir")
    result.add_argument("--input-space", choices=("native", "study"), default="native",
                        help="Default native MRI lets nnU-Net plan its own preprocessing; study is a separate comparison")
    result.add_argument("--allow-nonstandard-cohort", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> dict:
    args = parser().parse_args(argv)
    dataset_id = args.dataset_id if args.dataset_id is not None else (501 if args.dataset == "brats" else 502)
    if not 1 <= dataset_id <= 999:
        raise ValueError("Dataset ID must be between 1 and 999")
    source = args.brats_csv if args.dataset == "brats" else args.ucsf_manifest
    if source is None:
        raise ValueError("UCSF requires --ucsf-manifest from the existing paired experiment")
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = (brats_rows(source, args.brats_data_root) if args.dataset == "brats"
            else ucsf_rows(source, args.ucsf_prepared_root, args.ucsf_data_root, args.input_space))
    if args.dataset == "brats" and not args.allow_nonstandard_cohort:
        canonical_ids = pd.read_csv(REPO_ROOT / "tumourCSV.csv")["Brats20ID"].astype(str).tolist()
        if [row["case_id"] for row in rows] != canonical_ids:
            raise ValueError("BraTS IDs/order differ from the repository's frozen study CSV. "
                             "Remap source paths without changing IDs/order, or declare a separate cohort.")
    split = validate_cohort(rows, args.dataset, args.allow_nonstandard_cohort)
    work_dir = args.work_dir.resolve()
    dataset_name = f"Dataset{dataset_id:03d}_" + ("BraTS2020Study" if args.dataset == "brats" else "UCSFBMSRStudy")
    raw_dir, preprocessed_dir, results_dir = (work_dir / name for name in
                                              ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"))
    dataset_dir = raw_dir / dataset_name
    config = {"schema_version": 1, "dataset": args.dataset, "dataset_id": dataset_id,
              "dataset_name": dataset_name, "raw_dir": str(raw_dir),
              "dataset_dir": str(dataset_dir), "preprocessed_dir": str(preprocessed_dir),
              "results_dir": str(results_dir), "regions": ["WT", "TC", "ET"] if args.dataset == "brats" else ["BM"],
              "threshold": 0.33, "split": split, "source_manifest": str(source),
              "probability_channels": ["WT", "TC", "ET"] if args.dataset == "brats" else ["background", "BM"],
              "probability_activation": "sigmoid" if args.dataset == "brats" else "softmax",
              "probability_layout": "CZYX",
              "expected_probability_channels": 3 if args.dataset == "brats" else 2,
              "source_manifest_sha256": sha256_file(source),
              "allow_nonstandard_cohort": args.allow_nonstandard_cohort,
              "input_space": args.input_space,
              "geometry_note": "Native training geometry; separate fixed study-grid GT and prediction alignment" if args.input_space == "native" else
                               "Study input grid; UCSF cache has synthetic RAS origin 0",
              "split_seed": 10 if args.dataset == "brats" else None,
              "normalization_note": "Original intensities; only official planned normalization" if args.input_space == "native" else
                                    "Study-normalized inputs; official planned normalization is applied subsequently",
              "evaluation_space": "Original BraTS study crop or existing UCSF NPZ grid; identical GT for all methods",
              "test_labels_excluded_from_planning": True}
    signatures = {row["case_id"]: source_signature(row["source_files"]) for row in rows}
    protocol = {**config, "source_files": signatures}
    state_path = work_dir / "preparation_state.json"
    state = {"protocol": protocol, "completed": {}}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("protocol") != protocol:
            raise ValueError("Existing preparation belongs to different sources/splits/protocol; use a new work directory")
    elif dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise ValueError("Refusing to overwrite an existing dataset without a matching preparation state")
    print(json.dumps({"dataset": dataset_name, "counts": {k: len(v) for k, v in split.items()},
                      "patient_counts": dict(Counter({row['patient_group']: row['split'] for row in rows}.values())),
                      "work_dir": str(work_dir), "dry_run": args.dry_run}, indent=2), flush=True)
    if args.dry_run:
        return config
    for directory in (raw_dir, preprocessed_dir, results_dir, dataset_dir):
        directory.mkdir(parents=True, exist_ok=True)
    write_json(state_path, state)
    records = []
    for index, row in enumerate(rows, start=1):
        case_id = row["case_id"]
        completed = state["completed"].get(case_id)
        if completed:
            record = completed["record"]
            for path, digest in completed["export_sha256"].items():
                if not Path(path).is_file() or sha256_file(Path(path)) != digest:
                    raise ValueError(f"Previously exported file was changed or removed: {path}")
        else:
            record = export_case(row, args.dataset, dataset_dir, args.allow_nonstandard_cohort, args.input_space)
            paths = list(dict.fromkeys([*record["image_files"], record["label_file"], record["native_label_file"]]))
            state["completed"][case_id] = {"record": record,
                "export_sha256": {path: sha256_file(Path(path)) for path in paths}}
            write_json(state_path, state)
        records.append(record)
        print(f"Exported/verified {index}/{len(rows)}: {case_id} ({row['split']})", flush=True)
    description = dataset_description(args.dataset, len(split["train"]) + len(split["valid"]))
    write_json(dataset_dir / "dataset.json", description)
    write_json(work_dir / "frozen_split.json", split)
    write_json(work_dir / "splits_final.json", [{"train": split["train"], "val": split["valid"]}])
    config["cases"] = records
    config["dataset_json"] = description
    write_json(work_dir / "experiment.json", config)
    print(f"Prepared official dataset: {work_dir / 'experiment.json'}", flush=True)
    return config


if __name__ == "__main__":
    main()
