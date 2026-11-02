"""Check random initialization, legacy compatibility, and the unchanged early stop."""

from pathlib import Path
from types import SimpleNamespace
import runpy
import sys

import pandas as pd
import pytest
import torch

from training.initialization_guard import guard_full_initialization
from training.trainer import Trainer
import scripts.train_ucsf_baseline_full as ucsf


def protocol_args(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,split\na,train\nb,valid\n", encoding="utf-8")
    return dict(from_scratch=True, dataset="UCSF-BMSR", seed=55,
                manifest=manifest, epochs=200, lr=5e-4,
                model_settings={"fusion": "concat", "boundary_weight": 0.1})


def test_scratch_refuses_legacy_weights_without_modifying_them(tmp_path):
    args = protocol_args(tmp_path)
    directory = tmp_path / "old_full"
    directory.mkdir()
    checkpoint = directory / "best_model_90.pth"
    checkpoint.write_bytes(b"original-warm-start-weights")
    with pytest.raises(ValueError, match="Cannot certify"):
        guard_full_initialization(directory, **args)
    assert checkpoint.read_bytes() == b"original-warm-start-weights"
    assert not (directory / "initialization_protocol.json").exists()


@pytest.mark.parametrize("change", ["seed", "mode", "manifest", "lr"])
def test_scratch_run_cannot_silently_change_protocol(tmp_path, change):
    args = protocol_args(tmp_path)
    directory = tmp_path / "new_full"
    guard_full_initialization(directory, **args)
    guard_full_initialization(directory, **args)
    if change == "seed":
        args["seed"] = 42
    elif change == "mode":
        args["from_scratch"] = False
    elif change == "manifest":
        args["manifest"].write_text("id,split\na,valid\nb,train\n", encoding="utf-8")
    else:
        args["lr"] = 1e-3
    with pytest.raises(ValueError, match="protocol differs"):
        guard_full_initialization(directory, **args)


def test_legacy_warm_start_directory_is_still_accepted(tmp_path):
    args = protocol_args(tmp_path)
    args["from_scratch"] = False
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "best_model_20.pth").write_bytes(b"legacy")
    guard_full_initialization(directory, **args)
    assert not (directory / "initialization_protocol.json").exists()


class TinyFull(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv3d(4, kwargs.get("n_classes", 1), 1)
        self.multiscale_context = torch.nn.Module()
        self.multiscale_context.alpha = torch.nn.Parameter(torch.tensor(0.0))


class RecordingTrainer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.net = kwargs.get("net", kwargs.get("model"))
        self.model_type = str(kwargs.get("model_type", kwargs.get("checkpoint_dir")))
        self.optimizer = SimpleNamespace(param_groups=[{"lr": kwargs["lr"]}])
        self.epoch_value = 0
        self.losses = {"train": [], "valid": []}
        self.__class__.instances.append(self)

    def load_pretrain_model(self, path):
        raise AssertionError("A fresh scratch run must not load weights")

    def run(self, check_path):
        torch.save(self.net.state_dict(), Path(check_path) / "best_model_1.pth")


def args_for_ucsf(tmp_path, *, from_scratch=True):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("subject_id,split\n100001A,train\n", encoding="utf-8")
    return SimpleNamespace(seed=55, epochs=200, lr=5e-4,
                           manifest=manifest, from_scratch=from_scratch)


def test_ucsf_scratch_neither_requires_nor_loads_baseline(tmp_path, monkeypatch):
    args = args_for_ucsf(tmp_path)
    monkeypatch.setattr(ucsf, "ResUNetHFConcatBoundary", TinyFull)
    monkeypatch.setattr(ucsf, "make_trainer", lambda **kwargs: RecordingTrainer(**kwargs))

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Scratch Full tried to read the paired Baseline")

    monkeypatch.setattr(ucsf, "load_matching_baseline", forbidden_load)
    checkpoint = ucsf.train_full(args, tmp_path / "scratch", "cpu")
    assert checkpoint.is_file()
    trainer = RecordingTrainer.instances[-1]
    assert trainer.kwargs["boundary"] is True
    assert trainer.kwargs["epochs"] == 200
    assert trainer.kwargs["lr"] == 5e-4
    assert (checkpoint.parent / "initialization_protocol.json").is_file()
    assert (checkpoint.parent / "training_complete.json").is_file()


def test_ucsf_default_still_loads_baseline(tmp_path, monkeypatch):
    args = args_for_ucsf(tmp_path, from_scratch=False)
    monkeypatch.setattr(ucsf, "ResUNetHFConcatBoundary", TinyFull)
    monkeypatch.setattr(ucsf, "make_trainer", lambda **kwargs: RecordingTrainer(**kwargs))
    calls = []
    baseline = tmp_path / "baseline.pth"
    monkeypatch.setattr(ucsf, "load_matching_baseline", lambda model, path, device: (calls.append(path) or (1, 2)))
    ucsf.train_full(args, tmp_path / "warm", "cpu", baseline)
    assert calls == [baseline]


def test_ucsf_factory_preserves_early_stopping_and_accumulation(tmp_path, monkeypatch):
    monkeypatch.setattr(ucsf, "UCSFBoundaryTrainer", RecordingTrainer)
    trainer = ucsf.make_trainer(
        model=TinyFull(), criterion=torch.nn.Identity(),
        manifest=tmp_path / "manifest.csv", checkpoint_dir=tmp_path,
        epochs=200, lr=5e-4, boundary=True,
    )
    assert trainer.kwargs["early_stopping_patience"] == 25
    assert trainer.kwargs["min_delta"] == 1e-4
    assert trainer.kwargs["accumulation_steps"] == 4
    assert trainer.kwargs["batch_size"] == 1


def test_ucsf_cli_full_scratch_skips_completed_baseline_lookup(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("unused", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_ucsf_baseline_full.py", "--models", "full",
                        "--from-scratch", "--manifest", str(manifest),
                        "--output-root", str(tmp_path / "runs")])
    monkeypatch.setattr(ucsf, "read_manifest", lambda path: pd.DataFrame({
        "split": ["train", "valid", "test"], "patient_group": ["1", "2", "3"]}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ucsf, "completed_checkpoint", lambda path: pytest.fail("Unexpected paired Baseline lookup"))
    calls = []
    monkeypatch.setattr(ucsf, "train_full", lambda args, output, device, baseline: calls.append((args.from_scratch, baseline)))
    ucsf.main()
    assert calls == [(True, None)]


def test_brats_cli_scratch_skips_warm_start_and_retains_early_stop(tmp_path, monkeypatch):
    import models.resunet_hf_concat_boundary as full_models
    import importlib
    config_module = importlib.import_module("training.config")
    import training.trainer as trainer_module
    from matplotlib.figure import Figure

    csv_path = tmp_path / "tumourCSV.csv"
    csv_path.write_text("Brats20ID,path\nBraTS20_Training_001,/unused\n", encoding="utf-8")
    directory = tmp_path / "brats_scratch"
    monkeypatch.setattr(sys, "argv", ["train_hf_concat_boundary.py", "--from-scratch",
        "--multiscale_context_v2", "--boundary_weight", "0.1", "--seed", "55",
        "--csv", str(csv_path), "--checkpoint_dir", str(directory)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(full_models, "ResUNetHFConcatBoundary", TinyFull)
    monkeypatch.setattr(trainer_module, "Trainer", RecordingTrainer)
    monkeypatch.setattr(config_module, "check_exist", lambda path: pytest.fail("Unexpected Baseline checkpoint lookup"))
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    script = Path(__file__).resolve().parents[1] / "scripts/train_hf_concat_boundary.py"
    runpy.run_path(str(script), run_name="__main__")
    # HFConcatBoundaryTrainer is a subclass, so inspect its inherited list.
    trainer = RecordingTrainer.instances[-1]
    assert trainer.kwargs["path_to_csv"] == str(csv_path)
    assert trainer.kwargs["early_stopping_patience"] == 25
    assert trainer.kwargs["min_delta"] == 1e-4
    assert (directory / "initialization_protocol.json").is_file()


def test_existing_early_stop_stops_after_25_nonimproving_validations(tmp_path):
    trainer = object.__new__(Trainer)
    trainer.epoch_value = 0
    trainer.num_epochs = 200
    trainer.display_plot = False
    trainer.best_loss = float("inf")
    trainer.best_epoch = 0
    trainer.epochs_without_improvement = 0
    trainer.early_stopping_patience = 25
    trainer.min_delta = 1e-4
    trainer.model_type = str(tmp_path)
    trainer.net = torch.nn.Linear(1, 1)
    trainer.scheduler = SimpleNamespace(step=lambda loss: None)
    trainer._do_epoch = lambda epoch, phase: 0.5
    saved = []
    trainer._save_train_history = saved.append
    trainer.run(str(tmp_path))
    assert trainer.best_epoch == 1
    assert trainer.epochs_without_improvement == 25
    assert saved[-1] == 26
    assert (tmp_path / "best_model_1.pth").is_file()


@pytest.mark.parametrize("classes", [3, 1], ids=["brats", "ucsf"])
def test_real_formal_full_can_update_from_random_initialization(classes):
    from losses.enhanced import BCEDiceWithBoundaryLoss
    from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(55)
        model = ResUNetHFConcatBoundary(
            in_channels=4, n_classes=classes, n_channels=24,
            fusion="concat", multiscale_context_v2=True,
        ).train()
        assert model.multiscale_context.alpha.detach().item() == 0.0
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        image = torch.randn(1, 4, 16, 16, 16)
        target = torch.zeros(1, classes, 16, 16, 16)
        target[:, :, 4:12, 4:12, 4:12] = 1.0
        before = next(model.parameters()).detach().clone()
        segmentation, boundary = model(image)
        assert segmentation.shape == target.shape
        loss = BCEDiceWithBoundaryLoss(boundary_weight=0.1)((segmentation, boundary), target)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert model.multiscale_context.alpha.grad is not None
        optimizer.step()
        assert not torch.equal(before, next(model.parameters()).detach())
    finally:
        torch.set_num_threads(previous_threads)
