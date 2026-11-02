"""Exercise single-Full evaluation and shared HD95 scoring on fixed predictions."""

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.eval_ucsf_baseline_full as evaluation
from scripts.eval_official_nnunet import surface_metrics


class TinyPredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image):
        logits = image[:, :1] * self.scale
        return logits, logits


def study(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(evaluation, "build_model", lambda name: TinyPredictionModel())
    rows = []
    for index, (split, width) in enumerate([
        ("train", 3), ("train", 4), ("train", 5), ("test", 3), ("test", 4)
    ]):
        mask = np.zeros((1, 16, 16, 16), dtype=np.uint8)
        mask[:, 2:2+width, 2:2+width, 2:2+width] = 1
        image = np.full((4, 16, 16, 16), -3, dtype=np.float32)
        if index != 4:
            image[0][mask[0].astype(bool)] = 3
        path = tmp_path / f"case{index}.npz"
        np.savez_compressed(path, image=image, mask=mask,
                            spacing_dhw=np.array([1.5, 0.8, 0.8], dtype=np.float32))
        rows.append({"subject_id": f"{100001+index}A", "patient_group": str(100001+index),
                     "split": split, "prepared_path": str(path)})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    root = tmp_path / "scratch"
    full = root / "seed55/full"
    full.mkdir(parents=True)
    checkpoint = full / "best_model_3.pth"
    torch.save(TinyPredictionModel().state_dict(), checkpoint)
    marker = {"initialization": "from_scratch", "seed": 55,
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (full / "initialization_protocol.json").write_text(json.dumps(marker), encoding="utf-8")
    (full / "training_complete.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    return manifest, root, checkpoint


def test_full_only_cli_evaluates_once_and_packages_both_metric_profiles(tmp_path, monkeypatch):
    manifest, root, checkpoint = study(tmp_path, monkeypatch)
    original = checkpoint.read_bytes()
    output = tmp_path / "evaluation"
    calls = []
    evaluate = evaluation.evaluate_model

    def tracked(*args, **kwargs):
        calls.append(args[0])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_model", tracked)
    evaluation.main([
        "--manifest", str(manifest), "--training-root", str(root),
        "--models", "full", "--metric-profile", "both", "--output-dir", str(output),
    ])
    assert calls == ["full"]
    assert not (root / "seed55/baseline").exists()
    assert checkpoint.read_bytes() == original
    summary = pd.read_csv(output / "summary.csv").iloc[0]
    common = pd.read_csv(output / "summary_manuscript.csv").iloc[0]
    assert summary["n_test_scans"] == common["n_test_scans"] == 2
    assert summary["dice_mean"] == common["dice_mean"] == 0.5
    assert summary["hd95_mm_mean"] == 0
    assert summary["hd95_finite_cases"] == 1
    assert common["hd95_finite_cases"] == 2
    assert common["hd95_mm_mean"] > 0
    assert common["lesion_f1"] == summary["lesion_f1"]
    cases = pd.read_csv(output / "per_case.csv")
    assert cases["pred_empty"].sum() == 1
    assert not (output / "paired_case_dice.csv").exists()
    with zipfile.ZipFile(output / "results_for_table.zip") as archive:
        assert {"summary.csv", "summary.json", "summary_manuscript.csv", "evaluation.json"}.issubset(archive.namelist())
        assert "training/full/initialization_protocol.json" in archive.namelist()
        assert all(not name.endswith((".pth", ".npz", ".nii.gz")) for name in archive.namelist())


def test_paired_default_remains_legacy_and_writes_paired_cases(tmp_path, monkeypatch):
    manifest, root, checkpoint = study(tmp_path, monkeypatch)
    baseline = root / "seed55/baseline"
    baseline.mkdir()
    torch.save(TinyPredictionModel().state_dict(), baseline / "best_model_2.pth")
    output = tmp_path / "paired"
    evaluation.main(["--manifest", str(manifest), "--training-root", str(root),
                     "--output-dir", str(output)])
    summary = pd.read_csv(output / "summary.csv")
    assert summary["model"].tolist() == ["baseline", "full"]
    assert (output / "paired_case_dice.csv").is_file()
    assert not (output / "summary_manuscript.csv").exists()
    metadata = json.loads((output / "evaluation.json").read_text())
    assert metadata["protocol"]["summary_csv_profile"] == "legacy-ucsf"


def test_manuscript_profile_calls_official_surface_definition(tmp_path, monkeypatch):
    manifest, root, checkpoint = study(tmp_path, monkeypatch)
    output = tmp_path / "common"
    evaluation.main([
        "--manifest", str(manifest), "--training-root", str(root),
        "--models", "full", "--metric-profile", "manuscript", "--output-dir", str(output),
    ])
    cases = pd.read_csv(output / "per_case.csv")
    empty = np.zeros((16, 16, 16), dtype=bool)
    target = empty.copy()
    target[2:6, 2:6, 2:6] = True
    expected, _ = surface_metrics(empty, target, [1.5, 0.8, 0.8])
    assert cases.iloc[1]["hd95_mm"] == pytest.approx(expected)
    assert np.isfinite(cases["hd95_mm"]).all()


def test_evaluation_refuses_incomplete_or_changed_scratch_provenance(tmp_path, monkeypatch):
    manifest, root, checkpoint = study(tmp_path, monkeypatch)
    marker = checkpoint.parent / "training_complete.json"
    marker.unlink()
    with pytest.raises(ValueError, match="not marked completed"):
        evaluation.checkpoint_provenance(checkpoint, manifest, 55)
    marker.write_text('{"status": "completed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="seed differs"):
        evaluation.checkpoint_provenance(checkpoint, manifest, 42)
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest differs"):
        evaluation.checkpoint_provenance(checkpoint, manifest, 55)


def test_nonempty_output_from_another_protocol_is_not_overwritten(tmp_path, monkeypatch):
    manifest, root, checkpoint = study(tmp_path, monkeypatch)
    output = tmp_path / "old_eval"
    output.mkdir()
    summary = output / "summary.csv"
    summary.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="Nonempty evaluation"):
        evaluation.main(["--manifest", str(manifest), "--training-root", str(root),
                         "--models", "full", "--output-dir", str(output)])
    assert summary.read_text() == "original"
