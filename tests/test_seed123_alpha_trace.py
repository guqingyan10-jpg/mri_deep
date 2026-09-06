import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_seed123_alpha_trace.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_seed123_alpha_trace", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_seed123_trace_job_uses_new_directory_and_original_protocol():
    runner = _load_runner()
    output_root = ROOT / "synthetic-stability-output"
    expected_baseline = output_root / "seed123" / "baseline" / "best_model_17.pth"
    runner.find_completed_baseline = lambda _directory: expected_baseline
    job = runner.build_job(output_root, epochs=200, lr=5e-4)
    command = job.command()

    assert job.baseline_checkpoint == expected_baseline
    assert job.checkpoint_dir.name.endswith("_alpha_trace")
    assert job.checkpoint_dir != job.original_dir
    assert job.original_dir.name == "hf_concat_boundary_w0.1_multiscale_v2"
    assert command[command.index("--seed") + 1] == "123"
    assert command[command.index("--epochs") + 1] == "200"
    assert command[command.index("--lr") + 1] == "0.0005"
    assert command[command.index("--boundary_weight") + 1] == "0.1"
    assert command[command.index("--fusion") + 1] == "concat"
    assert "--multiscale_context_v2" in command
    assert "--from_scratch" not in command


def test_training_entrypoint_records_and_plots_alpha_each_epoch():
    source = (ROOT / "scripts" / "train_hf_concat_boundary.py").read_text(
        encoding="utf-8"
    )
    assert "def record_alpha_snapshot" in source
    assert "def _save_train_history" in source
    assert "self.record_alpha_snapshot(epoch)" in source
    assert '"alpha_history.csv"' in source
    assert '"alpha_learning_curve.{suffix}"' in source
    assert '("png", "pdf")' in source
    assert "trainer.record_alpha_snapshot(trainer.epoch_value)" in source
    assert "trainer.save_alpha_curve()" in source
