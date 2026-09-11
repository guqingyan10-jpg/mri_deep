import csv
from pathlib import Path

import numpy as np
import torch

from data.ucsf_bmsr import (
    UCSFPreparedDataset,
    assign_grouped_splits,
    get_ucsf_dataloader,
    patient_group,
    read_manifest,
)
from losses.basics import BCEDiceLoss
from losses.enhanced import BCEDiceWithBoundaryLoss
from models.resunet3d import ResUNet3d
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
from scripts.eval_ucsf_baseline_full import (
    classify_volume,
    derive_volume_strata,
    summarize_core,
)
from training.trainer import Trainer
from training.ucsf_trainer import UCSFTrainer


def test_grouped_split_is_deterministic_and_has_no_longitudinal_leakage():
    subject_ids = [
        f"{100000 + patient}{suffix}"
        for patient in range(30)
        for suffix in (("A", "B") if patient % 4 == 0 else ("A",))
    ]
    first = assign_grouped_splits(subject_ids, seed=10)
    second = assign_grouped_splits(list(reversed(subject_ids)), seed=10)
    assert first == second
    assert set(first.values()) == {"train", "valid", "test"}
    by_patient = {}
    for subject_id, split in first.items():
        by_patient.setdefault(patient_group(subject_id), set()).add(split)
    assert all(len(splits) == 1 for splits in by_patient.values())


def test_manifest_loader_uses_frozen_split_without_resplitting(tmp_path):
    rows = []
    for index, split in enumerate(("train", "valid", "test"), start=1):
        subject_id = f"10000{index}A"
        case_path = tmp_path / f"{subject_id}.npz"
        np.savez_compressed(
            case_path,
            image=np.zeros((4, 16, 16, 16), dtype=np.float16),
            mask=np.zeros((1, 16, 16, 16), dtype=np.uint8),
            spacing_dhw=np.ones(3, dtype=np.float32),
        )
        rows.append({
            "subject_id": subject_id,
            "patient_group": patient_group(subject_id),
            "split": split,
            "prepared_path": str(case_path),
        })
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert len(read_manifest(manifest)) == 3
    for split, expected_id in zip(("train", "valid", "test"), ("100001A", "100002A", "100003A")):
        loader = get_ucsf_dataloader(
            UCSFPreparedDataset, str(manifest), split,
            batch_size=1, num_workers=0,
        )
        batch = next(iter(loader))
        assert batch["Id"] == [expected_id]
        assert tuple(batch["image"].shape) == (1, 4, 16, 16, 16)
        assert tuple(batch["mask"].shape) == (1, 1, 16, 16, 16)


def test_binary_baseline_and_full_have_matching_task_shapes_and_losses():
    image = torch.randn(1, 4, 16, 16, 16)
    target = torch.zeros(1, 1, 16, 16, 16)
    baseline = ResUNet3d(in_channels=4, n_classes=1, n_channels=8).eval()
    full = ResUNetHFConcatBoundary(
        in_channels=4,
        n_classes=1,
        n_channels=8,
        fusion="concat",
        multiscale_context_v2=True,
    ).eval()
    with torch.no_grad():
        baseline_logits = baseline(image)
        full_logits, boundary_logits = full(image)
    assert baseline_logits.shape == target.shape
    assert full_logits.shape == target.shape
    assert boundary_logits.shape == target.shape
    assert torch.isfinite(BCEDiceLoss()(baseline_logits, target))
    assert torch.isfinite(
        BCEDiceWithBoundaryLoss(boundary_weight=0.1)(
            (full_logits, boundary_logits), target
        )
    )


def test_ucsf_trainer_keeps_only_latest_recovery_checkpoint(tmp_path, monkeypatch):
    for epoch in (1, 2):
        (tmp_path / f"last_epoch_model_{epoch}.pth").write_bytes(b"old")

    def fake_save_train_history(self, epoch):
        (Path(self.model_type) / f"last_epoch_model_{epoch}.pth").write_bytes(b"new")

    monkeypatch.setattr(Trainer, "_save_train_history", fake_save_train_history)
    trainer = object.__new__(UCSFTrainer)
    trainer.model_type = str(tmp_path)
    trainer._save_train_history(3)

    assert sorted(path.name for path in tmp_path.glob("last_epoch_model_*.pth")) == [
        "last_epoch_model_3.pth"
    ]


def test_ucsf_volume_strata_are_fitted_from_training_values():
    strata = derive_volume_strata([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    assert strata == {
        "small": (0.0, 20.0),
        "medium": (20.0, 40.0),
        "large": (40.0, None),
    }
    assert classify_volume(20.0, strata) == "small"
    assert classify_volume(21.0, strata) == "medium"
    assert classify_volume(41.0, strata) == "large"


def test_ucsf_primary_summary_is_limited_to_brats_aligned_metrics(tmp_path):
    cases = [{
        "dice": 0.8,
        "hd95_mm": 2.0,
        "tp_lesions": 2,
        "fp_lesions": 1,
        "fn_lesions": 1,
    }]
    lesions = [
        {"volume_stratum": "small", "gt_anchored_dice": 0.6},
        {"volume_stratum": "small", "gt_anchored_dice": 0.0},
        {"volume_stratum": "large", "gt_anchored_dice": 0.9},
    ]
    summary = summarize_core("baseline", tmp_path / "best_model_1.pth", cases, lesions)
    metric_keys = {
        "dice_mean",
        "hd95_mm_mean",
        "lesion_gt_anchored_dice",
        "small_lesion_gt_anchored_dice",
        "lesion_f1",
    }
    assert metric_keys.issubset(summary)
    assert summary["lesion_gt_anchored_dice"] == 0.5
    assert summary["small_lesion_gt_anchored_dice"] == 0.3
    assert not {"surface_dice", "lesion_precision", "lesion_recall"}.intersection(summary)


def test_ucsf_documentation_disables_five_fold_and_pins_formal_full():
    root = Path(__file__).resolve().parents[1]
    training = (root / "scripts" / "train_ucsf_baseline_full.py").read_text(encoding="utf-8")
    documentation = (root / "docs" / "UCSF_EXTERNAL_BASELINE_FULL.md").read_text(encoding="utf-8")
    assert 'n_classes=1' in training
    assert 'boundary_weight=0.1' in training
    assert 'multiscale_context_v2=True' in training
    assert 'fusion="concat"' in training
    assert "不使用五折" in documentation
