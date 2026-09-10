"""Prepare UCSF-BMSR for paired ResUNet Baseline/Full training.

The source data are never modified. Every modality is reoriented to canonical
RAS, the complete field of view is resized to the BraTS tensor size, each
channel is min-max normalized, and the binary metastasis label is resized with
nearest-neighbor interpolation. Longitudinal scans stay in one patient split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ucsf_bmsr import assign_grouped_splits, patient_group


MODALITIES = ("T1post", "T1pre", "FLAIR", "subtraction")
TARGET_DHW = (100, 170, 170)


def source_paths(subject_dir: Path) -> tuple[list[Path], Path]:
    subject_id = subject_dir.name
    images = [subject_dir / f"{subject_id}_{name}.nii.gz" for name in MODALITIES]
    label = subject_dir / f"{subject_id}_seg.nii.gz"
    return images, label


def discover_cases(data_root: Path) -> list[Path]:
    cases = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not cases:
        raise FileNotFoundError(f"No subject directories found in {data_root}")
    missing = []
    for case in cases:
        images, label = source_paths(case)
        missing.extend(str(path) for path in [*images, label] if not path.is_file())
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required UCSF files; examples: {missing[:10]}"
        )
    return cases


def load_canonical(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def resize_dhw(array_dhw: np.ndarray, target_dhw: tuple[int, int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(array_dhw)).float()[None, None]
    kwargs = {"size": target_dhw, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor, **kwargs)[0, 0]
    return resized.numpy()


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - minimum) / (maximum - minimum)).astype(np.float32)


def prepare_case(subject_dir: Path, destination: Path, target_dhw: tuple[int, int, int]) -> dict:
    image_paths, label_path = source_paths(subject_dir)
    reference = load_canonical(image_paths[0])
    reference_shape = tuple(int(value) for value in reference.shape[:3])
    reference_affine = reference.affine
    channels = []
    for path in image_paths:
        image = load_canonical(path)
        if tuple(image.shape[:3]) != reference_shape:
            raise ValueError(f"Modality shape mismatch for {subject_dir.name}: {path}")
        if not np.allclose(image.affine, reference_affine, rtol=1e-4, atol=1e-3):
            raise ValueError(f"Modality affine mismatch for {subject_dir.name}: {path}")
        xyz = np.asarray(image.dataobj, dtype=np.float32)
        dhw = np.transpose(xyz, (2, 1, 0))
        channels.append(normalize_minmax(resize_dhw(dhw, target_dhw, "trilinear")))

    label_image = load_canonical(label_path)
    if tuple(label_image.shape[:3]) != reference_shape:
        raise ValueError(f"Label shape mismatch for {subject_dir.name}: {label_path}")
    if not np.allclose(label_image.affine, reference_affine, rtol=1e-4, atol=1e-3):
        raise ValueError(f"Label affine mismatch for {subject_dir.name}: {label_path}")
    label_xyz = np.asarray(label_image.dataobj)
    labels = np.unique(label_xyz)
    if not set(labels.tolist()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Expected binary _seg label for {subject_dir.name}, got {labels}")
    label_dhw = np.transpose((label_xyz > 0).astype(np.float32), (2, 1, 0))
    mask = (resize_dhw(label_dhw, target_dhw, "nearest") > 0.5).astype(np.uint8)[None]

    zooms_xyz = np.asarray(reference.header.get_zooms()[:3], dtype=np.float64)
    target_xyz = np.asarray(target_dhw[::-1], dtype=np.float64)
    effective_spacing_xyz = zooms_xyz * np.asarray(reference_shape) / target_xyz
    spacing_dhw = effective_spacing_xyz[::-1].astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        image=np.stack(channels).astype(np.float16),
        mask=mask,
        spacing_dhw=spacing_dhw,
    )
    return {
        "original_shape_xyz": "x".join(map(str, reference_shape)),
        "original_spacing_xyz": "x".join(f"{value:.6g}" for value in zooms_xyz),
        "prepared_shape_dhw": "x".join(map(str, target_dhw)),
        "effective_spacing_dhw": "x".join(f"{value:.6g}" for value in spacing_dhw),
        "label_voxels_original": int((label_xyz > 0).sum()),
        "label_voxels_prepared": int(mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/UCSF_BrainMetastases_v1.3/UCSF_BrainMetastases_TRAIN"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/root/autodl-tmp/ucsf_bmsr_prepared"),
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--target-shape", nargs=3, type=int, default=TARGET_DHW,
                        metavar=("D", "H", "W"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = discover_cases(args.data_root)
    subject_ids = [case.name for case in cases]
    assignments = assign_grouped_splits(subject_ids, seed=args.seed)
    split_counts = Counter(assignments.values())
    group_counts = Counter(
        (assignments[subject_id], patient_group(subject_id))
        for subject_id in subject_ids
    )
    groups_per_split = Counter(split_name for split_name, _ in group_counts)
    print(f"Found {len(cases)} labeled scans from {len(set(patient_group(x) for x in subject_ids))} patients")
    print(f"Scan split counts: {dict(split_counts)}")
    print(f"Patient split counts: {dict(groups_per_split)}")
    print(f"Modalities: {', '.join(MODALITIES)}; target DHW={tuple(args.target_shape)}")
    if args.dry_run:
        return

    prepared_dir = args.output_root / "cases"
    rows = []
    for number, subject_dir in enumerate(cases, start=1):
        subject_id = subject_dir.name
        destination = (prepared_dir / f"{subject_id}.npz").resolve()
        if destination.is_file() and not args.force:
            with np.load(destination, allow_pickle=False) as cached:
                cached_mask = np.asarray(cached["mask"])
                spacing = np.asarray(cached["spacing_dhw"])
                shape = tuple(int(value) for value in cached_mask.shape[1:])
            details = {
                "original_shape_xyz": "cached",
                "original_spacing_xyz": "cached",
                "prepared_shape_dhw": "x".join(map(str, shape)),
                "effective_spacing_dhw": "x".join(f"{value:.6g}" for value in spacing),
                "label_voxels_original": "cached",
                "label_voxels_prepared": int(cached_mask.sum()),
            }
        else:
            details = prepare_case(subject_dir, destination, tuple(args.target_shape))
        rows.append({
            "subject_id": subject_id,
            "patient_group": patient_group(subject_id),
            "split": assignments[subject_id],
            "prepared_path": str(destination),
            "source_dir": str(subject_dir.resolve()),
            **details,
        })
        if number % 10 == 0 or number == len(cases):
            print(f"Prepared {number}/{len(cases)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "modalities": list(MODALITIES),
        "label": "<SubjectID>_seg.nii.gz; foreground value 1",
        "target_shape_dhw": list(args.target_shape),
        "normalization": "per-modality min-max after whole-FOV resize",
        "orientation": "canonical RAS before resize",
        "split_seed": args.seed,
        "split_method": "patient-grouped fixed 70/20/10, matching BraTS random_state=10 logic",
        "scan_counts": dict(split_counts),
        "patient_counts": dict(groups_per_split),
    }
    (args.output_root / "preparation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
