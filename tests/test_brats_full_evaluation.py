"""Check the new scratch Full row without requiring any other model weights."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts import eval_brats_full as evaluation


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image):
        logits = image[:, :3] * self.scale
        return logits, logits[:, :1]


class TinyCases(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        target = torch.zeros(3, 16, 16, 16)
        target[:, 4:8, 4:8, 4:8] = 1
        image = torch.full((4, 16, 16, 16), -3.0)
        if index == 0:
            image[:3][target.bool()] = 3.0
        return {"image": image, "mask": target, "Id": f"case{index}"}


def test_legacy_metrics_use_existing_surface_rules_and_pooled_f1(tmp_path):
    cases, lesions = evaluation.evaluate_loaded_model(
        TinyModel(), DataLoader(TinyCases(), batch_size=1), torch.device("cpu"), "legacy-brats", 1.0)
    summary, _ = evaluation.summarize_model(tmp_path / "best_model_1.pth", 55, "legacy-brats", cases, lesions)
    assert summary["Macro_Dice_mean"] == summary["ET_Dice_mean"] == 0.5
    assert summary["ET_HD95_mean"] == 0
    assert summary["Boundary_Dice_mean"] == 1
    assert summary["ET_HD95_finite_cases"] == summary["Boundary_Dice_finite_cases"] == 1
    assert summary["Lesion_F1"] == pytest.approx(2 / 3)
    assert (summary["tp_lesions"], summary["fp_lesions"], summary["fn_lesions"]) == (1, 0, 1)
    assert summary["pred_empty_cases"] == 1


def test_manuscript_profile_counts_empty_case_with_same_predictions(tmp_path):
    cases, lesions = evaluation.evaluate_loaded_model(
        TinyModel(), DataLoader(TinyCases(), batch_size=1), torch.device("cpu"), "manuscript", 1.0)
    summary, _ = evaluation.summarize_model(tmp_path / "best_model_1.pth", 55, "manuscript", cases, lesions)
    assert summary["ET_HD95_mean"] == pytest.approx(np.linalg.norm([16, 16, 16]) / 2)
    assert summary["Boundary_Dice_mean"] == 0.5
    assert summary["ET_HD95_finite_cases"] == 2


def make_run(tmp_path):
    directory = tmp_path / "scratch"
    directory.mkdir()
    checkpoint = directory / "best_model_53.pth"
    torch.save(TinyModel().state_dict(), checkpoint)
    initialization = {"initialization": "from_scratch", "dataset": "BraTS2020", "seed": 55,
        "manifest_sha256": evaluation.file_sha256(evaluation.ROOT / "tumourCSV.csv"),
        "model_settings": {"fusion": "concat", "boundary_weight": 0.1,
                           "multiscale_context_v2": True, "base_channels": 24}}
    (directory / "initialization_protocol.json").write_text(json.dumps(initialization), encoding="utf-8")
    return directory, checkpoint


def fake_evaluation(model, dataloader, device, profile, tolerance):
    cases = [{"model": evaluation.MODEL_NAME, "case_id": case_id, "WT_Dice": 1.0, "TC_Dice": 1.0,
              "ET_Dice": 1.0, "Macro_Dice": 1.0, "ET_HD95": 0.0, "Boundary_Dice": 1.0,
              "gt_empty": False, "pred_empty": False, "gt_lesions": 0,
              "pred_lesions": 0, "tp_lesions": 0, "fp_lesions": 0, "fn_lesions": 0}
             for case_id in dataloader.dataset.df["Brats20ID"].astype(str)]
    lesions = []
    index = 0
    for stratum, count, size in (("small", 31, 27), ("medium", 36, 64), ("large", 26, 5000)):
        for _ in range(count):
            case = cases[index % len(cases)]
            case["gt_lesions"] += 1
            case["pred_lesions"] += 1
            case["tp_lesions"] += 1
            lesions.append({"model": evaluation.MODEL_NAME, "case_id": case["case_id"],
                            "gt_component_index": index, "gt_voxels": size, "stratum": stratum,
                            "detected": True, "matched_dice": 1.0, "gt_anchored_dice": 1.0})
            index += 1
    return cases, lesions


def test_cli_selects_only_scratch_full_and_packages_five_metrics(tmp_path, monkeypatch):
    directory, checkpoint = make_run(tmp_path)
    original = checkpoint.read_bytes()
    calls = []
    monkeypatch.setattr(evaluation, "load_full", lambda path, device: calls.append(path) or TinyModel())
    monkeypatch.setattr(evaluation, "evaluate_loaded_model", fake_evaluation)
    output = tmp_path / "evaluation"
    evaluation.main(["--checkpoint-dir", str(directory), "--output-dir", str(output), "--device", "cpu"])
    assert calls == [checkpoint.resolve()]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_test_scans"] == 37
    assert summary["seed"] == 55 and summary["model_id"] == "full_scratch"
    assert summary["Lesion_F1"] == 1.0
    assert checkpoint.read_bytes() == original
    assert len(pd.read_csv(output / "per_case.csv")) == 37
    assert len(pd.read_csv(output / "per_gt_lesion.csv")) == 93
    with zipfile.ZipFile(output / "results_for_table.zip") as archive:
        assert {"comparison_row.csv", "summary.csv", "summary.json", "per_case.csv"}.issubset(archive.namelist())
        assert "training/full/initialization_protocol.json" in archive.namelist()
        assert all(not name.endswith((".pth", ".npz", ".nii.gz")) for name in archive.namelist())


def test_dry_run_reads_no_mri_and_loads_no_weights(tmp_path, monkeypatch):
    directory, _ = make_run(tmp_path)
    monkeypatch.setattr(evaluation, "load_full", lambda *args: pytest.fail("dry run loaded model"))
    output = tmp_path / "dry"
    evaluation.main(["--checkpoint-dir", str(directory), "--output-dir", str(output), "--dry-run"])
    assert not output.exists()


def test_wrong_seed_or_initialization_is_refused(tmp_path):
    directory, checkpoint = make_run(tmp_path)
    csv_path = evaluation.ROOT / "tumourCSV.csv"
    with pytest.raises(ValueError, match="seed or training CSV"):
        evaluation.initialization_record(checkpoint, csv_path, 42)
    marker = directory / "initialization_protocol.json"
    record = json.loads(marker.read_text())
    record["initialization"] = "warm_start"
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="not a from-scratch"):
        evaluation.initialization_record(checkpoint, csv_path, 55)


def test_formal_full_loads_strictly_without_baseline(tmp_path):
    from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
    model = ResUNetHFConcatBoundary(in_channels=4, n_classes=3, n_channels=24,
                                    fusion="concat", multiscale_context_v2=True)
    checkpoint = tmp_path / "best_model_2.pth"
    torch.save(model.state_dict(), checkpoint)
    loaded = evaluation.load_full(checkpoint, torch.device("cpu"))
    assert loaded.multiscale_context_v2_enabled
    assert len(loaded.state_dict()) == len(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
