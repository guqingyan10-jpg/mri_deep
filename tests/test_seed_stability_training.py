import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "scripts" / "run_seed_stability.py"
    spec = importlib.util.spec_from_file_location("run_seed_stability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stability_runner_builds_seed_isolated_warm_start_jobs():
    runner_path = ROOT / "scripts" / "run_seed_stability.py"
    assert runner_path.exists(), "stability runner is missing"
    runner = _load_runner()

    output_root = ROOT / "stability-output"
    jobs = runner.build_jobs(seeds=[42], output_root=output_root, epochs=200, lr=5e-4)

    assert [job.name for job in jobs] == [
        "baseline",
        "edge_laplacian_concat",
        "hf_concat_boundary_w0.1",
        "hf_concat_boundary_w0.05",
    ]
    assert jobs[0].checkpoint_dir == output_root / "seed42" / "baseline"
    assert all(job.checkpoint_dir.parent == output_root / "seed42" for job in jobs)
    assert all(job.seed == 42 for job in jobs)

    baseline_path = str(jobs[0].checkpoint_dir / "best_model_19.pth")
    for job in jobs[1:]:
        assert "--baseline_checkpoint" in job.command(baseline_path)
        assert baseline_path in job.command(baseline_path)


def test_baseline_entrypoint_has_explicit_seed_and_checkpoint_directory():
    source = (ROOT / "scripts" / "train_baseline_bcedice.py").read_text(encoding="utf-8")

    assert "BCEDiceLoss" in source
    assert '"--seed"' in source or "'--seed'" in source
    assert '"--checkpoint_dir"' in source or "'--checkpoint_dir'" in source
    assert "check_exist_last(CHECKPOINT_DIR)" in source
    assert "training_complete.json" in source
    assert "os.remove(completion_marker)" in source


def test_edge_and_hf_entrypoints_accept_explicit_seed_and_baseline_checkpoint():
    for name in ("train_v2_edge.py", "train_hf_concat_boundary.py"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        argument_values = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }

        assert "--seed" in argument_values
        assert "--checkpoint_dir" in argument_values
        assert "--baseline_checkpoint" in argument_values
        assert "best_model_" in source


def test_runner_refuses_baseline_without_completion_marker(tmp_path):
    runner = _load_runner()
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "best_model_10.pth").write_bytes(b"checkpoint")

    try:
        runner.find_completed_baseline(baseline_dir)
    except FileNotFoundError as exc:
        assert "training_complete.json" in str(exc)
    else:
        raise AssertionError("an unmarked baseline must not unlock derived models")


def test_runner_accepts_marked_baseline_with_best_checkpoint(tmp_path):
    runner = _load_runner()
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    best = baseline_dir / "best_model_10.pth"
    best.write_bytes(b"checkpoint")
    (baseline_dir / "training_complete.json").write_text(
        json.dumps({"status": "completed", "best_checkpoint": str(best)}),
        encoding="utf-8",
    )

    assert runner.find_completed_baseline(baseline_dir) == best
