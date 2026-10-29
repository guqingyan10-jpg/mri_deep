import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from data.dataset import BratsDataset, get_dataloader
from scripts import prepare_official_nnunet as prepare


def make_ucsf_manifest(tmp_path, shape=(4, 5, 6)):
    rows = []
    for index, split in enumerate(("train", "valid", "test"), start=1):
        subject = f"10000{index}A"
        path = tmp_path / f"{subject}.npz"
        image = np.arange(4 * np.prod(shape), dtype=np.float32).reshape((4, *shape))
        mask = np.zeros((1, *shape), dtype=np.uint8)
        mask[0, 1, 2, 3] = 1
        np.savez_compressed(path, image=image.astype(np.float16), mask=mask,
                            spacing_dhw=np.asarray([3.0, 2.0, 1.0]))
        rows.append({"subject_id": subject, "patient_group": str(100000 + index),
                     "split": split, "prepared_path": str(path)})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def ucsf_args(manifest, work):
    return ["--dataset", "ucsf", "--ucsf-manifest", str(manifest),
            "--work-dir", str(work), "--input-space", "study", "--allow-nonstandard-cohort"]


def test_ucsf_export_preserves_exact_cache_and_excludes_test_labels_from_planning(tmp_path):
    manifest = make_ucsf_manifest(tmp_path)
    work = tmp_path / "official"
    experiment = prepare.main(ucsf_args(manifest, work))
    assert experiment["regions"] == ["BM"]
    assert experiment["dataset_json"]["labels"] == {"background": 0, "BM": 1}
    assert "regions_class_order" not in experiment["dataset_json"]
    assert experiment["probability_channels"] == ["background", "BM"]
    raw = Path(experiment["dataset_dir"])
    assert len(list((raw / "imagesTr").glob("*.nii.gz"))) == 8
    assert len(list((raw / "labelsTr").glob("*.nii.gz"))) == 2
    assert not (raw / "labelsTr/100003A.nii.gz").exists()
    assert (raw / "labelsTs_reference/100003A.nii.gz").exists()
    frozen = json.loads((work / "splits_final.json").read_text())
    assert frozen == [{"train": ["100001A"], "val": ["100002A"]}]
    with np.load(tmp_path / "100003A.npz") as original:
        for channel, path in enumerate(experiment["cases"][2]["image_files"]):
            image = nib.load(path)
            np.testing.assert_array_equal(np.asarray(image.dataobj).transpose(2, 1, 0),
                                          original["image"][channel].astype(np.float32))
            np.testing.assert_allclose(nib.affines.voxel_sizes(image.affine), [1, 2, 3])
        gt = nib.load(experiment["cases"][2]["label_file"])
        np.testing.assert_array_equal(np.asarray(gt.dataobj).transpose(2, 1, 0), original["mask"][0])
    assert prepare.main(ucsf_args(manifest, work)) == experiment


def test_preparation_dry_run_creates_no_output(tmp_path):
    manifest = make_ucsf_manifest(tmp_path)
    work = tmp_path / "dry"
    result = prepare.main([*ucsf_args(manifest, work), "--dry-run"])
    assert result["split"]["test"] == ["100003A"]
    assert not work.exists()


def test_preparation_rejects_changed_source_and_corrupt_exports(tmp_path):
    manifest = make_ucsf_manifest(tmp_path)
    work = tmp_path / "official"
    experiment = prepare.main(ucsf_args(manifest, work))
    Path(experiment["cases"][0]["image_files"][0]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed or removed"):
        prepare.main(ucsf_args(manifest, work))
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="different sources/splits/protocol"):
        prepare.main(ucsf_args(manifest, work))


def test_nonstandard_cohort_is_explicit_and_patient_leakage_is_never_allowed(tmp_path):
    manifest = make_ucsf_manifest(tmp_path)
    with pytest.raises(ValueError, match="cohort mismatch"):
        prepare.main(["--dataset", "ucsf", "--ucsf-manifest", str(manifest),
                      "--work-dir", str(tmp_path / "official"), "--input-space", "study"])
    rows = [{"case_id": "100001A", "patient_group": "100001", "split": "train"},
            {"case_id": "100001B", "patient_group": "100001", "split": "test"},
            {"case_id": "100002A", "patient_group": "100002", "split": "valid"}]
    with pytest.raises(ValueError, match="Patient leakage"):
        prepare.validate_cohort(rows, "ucsf", True)


def test_ucsf_rejects_invalid_mask_or_spacing(tmp_path):
    source = tmp_path / "bad.npz"
    row = {"case_id": "100001A", "source_files": [source]}
    np.savez(source, image=np.zeros((4, 4, 5, 6)), mask=np.full((1, 4, 5, 6), 4),
             spacing_dhw=[1, 1, 1])
    with pytest.raises(ValueError, match="binary"):
        prepare.ucsf_arrays(row, True)
    np.savez(source, image=np.zeros((4, 4, 5, 6)), mask=np.zeros((1, 4, 5, 6)),
             spacing_dhw=[1, 0, 1])
    with pytest.raises(ValueError, match="spacing"):
        prepare.ucsf_arrays(row, True)


def test_brats_regions_preserve_original_label_semantics():
    config = prepare.dataset_description("brats", 331)
    assert config["labels"] == {"background": 0, "WT": [1, 2, 3], "TC": [1, 3], "ET": 3}
    assert config["regions_class_order"] == [2, 1, 3]
    assert config["numTraining"] == 331


def test_official_label_manager_round_trip_preserves_brats_regions_and_ucsf_binary_task():
    pytest.importorskip("nnunetv2")
    from nnunetv2.utilities.label_handling.label_handling import LabelManager

    config = prepare.dataset_description("brats", 331)
    manager = LabelManager(config["labels"], config["regions_class_order"])
    assert manager.has_regions
    probabilities = np.asarray([[0, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]], dtype=np.float32)[:, None, None]
    segmentation = manager.convert_probabilities_to_segmentation(probabilities)
    np.testing.assert_array_equal(segmentation.ravel(), [0, 1, 2, 3])

    binary = prepare.dataset_description("ucsf", 415)
    binary_manager = LabelManager(binary["labels"], binary.get("regions_class_order"))
    assert not binary_manager.has_regions
    assert binary_manager.num_segmentation_heads == 2
    binary_probs = binary_manager.apply_inference_nonlin(np.zeros((2, 2, 2, 2), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(binary_probs).sum(axis=0), 1)


def test_source_geometry_mismatch_is_rejected():
    reference = nib.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4))
    changed = nib.Nifti1Image(np.zeros((2, 3, 4)), np.diag([2, 1, 1, 1]))
    with pytest.raises(ValueError, match="affine mismatch"):
        prepare.check_image_geometry(changed, reference, Path("bad.nii.gz"))


def test_brats_split_reuses_csv_order_and_existing_dataloader(tmp_path, monkeypatch):
    csv_path = tmp_path / "brats.csv"
    ids = [f"BraTS20_Training_{value:03d}" for value in range(1, 370) if value != 355]
    # Reverse deliberately: the original CSV's order is part of the split.
    pd.DataFrame({"Brats20ID": ids[::-1], "path": [str(tmp_path)] * 368}).to_csv(csv_path, index=False)
    monkeypatch.setattr(prepare, "resolve_nifti", lambda directory, stem: directory / (stem + ".nii.gz"))
    rows = prepare.brats_rows(csv_path, None)
    split = prepare.validate_cohort(rows, "brats", False)
    assert {name: len(values) for name, values in split.items()} == {"train": 257, "valid": 74, "test": 37}
    for phase in ("train", "valid", "test"):
        reference = get_dataloader(BratsDataset, str(csv_path), phase, num_workers=0)
        assert split[phase] == reference.dataset.df["Brats20ID"].tolist()


def test_brats_crop_preserves_voxel_values_label_mapping_and_world_origin(tmp_path):
    shape = (210, 210, 120)
    source = np.broadcast_to(np.arange(shape[0], dtype=np.int16)[:, None, None], shape)
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    affine[:3, 3] = [5, 6, 7]
    mri = tmp_path / "image.nii.gz"
    label_file = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(source, affine), mri)
    label = np.zeros(shape, dtype=np.uint8)
    label[40, 40, 20], label[41, 40, 20], label[42, 40, 20] = 1, 2, 4
    nib.save(nib.Nifti1Image(label, affine), label_file)
    row = {"case_id": "BraTS20_Training_001", "split": "test",
           "patient_group": "BraTS20_Training_001", "source_files": [mri] * 4 + [label_file]}
    image, target, prepared_affine = prepare.brats_arrays(row)
    assert image.shape == (4, 170, 170, 100)
    assert target.shape == (170, 170, 100)
    np.testing.assert_array_equal(target[:3, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(image[0, :, 0, 0], np.arange(170) / 169)
    np.testing.assert_allclose(prepared_affine[:3, 3], [45, 86, 67])
    record = prepare.export_case(row, "brats", tmp_path / "export", False)
    assert record["shape_dhw"] == [100, 170, 170]
    assert record["spacing_dhw"] == [3.0, 2.0, 1.0]
    assert Path(record["label_file"]).parent.name == "labelsTs_reference"
