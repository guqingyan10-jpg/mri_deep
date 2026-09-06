import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_multiseed_typical_case_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("multiseed_typical_cases", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_default_triplets_keep_seed55_main_protocol_and_other_stability_runs():
    triplets = MODULE.default_seed_triplets(
        Path("/stability"),
        Path("/seed55/baseline"),
        Path("/seed55/lhfc"),
        Path("/seed55/full"),
    )

    assert [item.seed for item in triplets] == [42, 55, 123]
    assert triplets[0].full == Path(
        "/stability/seed42/hf_concat_boundary_w0.1_multiscale_v2"
    )
    assert triplets[1].protocol == "main_experiment"
    assert triplets[1].full == Path("/seed55/full")
    assert triplets[2].protocol == "stability_runner"


def test_seed123_supplemental_checkpoint_is_only_used_when_explicitly_overridden():
    override = Path("/supplemental/seed123_alpha_trace")
    triplets = MODULE.default_seed_triplets(
        Path("/stability"), Path("/b"), Path("/l"), Path("/f"), override
    )

    assert triplets[2].full == override


def test_selector_command_writes_each_seed_to_its_own_directory():
    triplet = MODULE.SeedModelTriplet(
        seed=42,
        baseline=Path("/models/baseline"),
        lhfc=Path("/models/lhfc"),
        full=Path("/models/full"),
        protocol="stability_runner",
    )
    command = triplet.selector_command(
        Path("results"), Path("strata.json"), Path("cases.csv")
    )

    assert str(MODULE.SELECTOR) in command
    assert command[command.index("--output-dir") + 1] == str(
        Path("results") / "seed42"
    )
    assert command[command.index("--baseline-checkpoint") + 1] == str(
        triplet.baseline
    )


def test_completed_seed_outputs_are_detected_for_resume():
    with patch.object(Path, "is_file", return_value=True):
        assert MODULE.seed_output_complete(Path("results/seed42"))
    with patch.object(Path, "is_file", side_effect=[True, True, True, False]):
        assert not MODULE.seed_output_complete(Path("results/seed123"))


def _small_row(seed, case_id, gt_index, baseline, full):
    return {
        "seed": seed,
        "case_id": case_id,
        "gt_index": gt_index,
        "gt_id": gt_index + 1,
        "gt_size": 20 + gt_index,
        "baseline_detected": True,
        "full_detected": True,
        "baseline_lesion_dice": baseline,
        "full_lesion_dice": full,
        "small_lesion_dice_gain": full - baseline,
    }


def test_small_lesion_ranking_prioritizes_same_lesion_improved_in_all_seeds():
    rows = []
    for seed in MODULE.SEEDS:
        rows.append(_small_row(seed, "consistent", 0, 0.20, 0.40))
    rows.extend(
        [
            _small_row(42, "mixed", 1, 0.10, 0.70),
            _small_row(55, "mixed", 1, 0.10, 0.60),
            _small_row(123, "mixed", 1, 0.40, 0.30),
        ]
    )

    ranked = MODULE.aggregate_small_lesions(pd.DataFrame(rows))

    assert ranked.iloc[0]["case_id"] == "consistent"
    assert bool(ranked.iloc[0]["all_seeds_improved"])
    assert ranked.iloc[0]["improved_seeds"] == 3
    assert ranked.iloc[1]["improved_seeds"] == 2


def _boundary_row(seed, case_id, hd95_gain, bd_gain):
    return {
        "seed": seed,
        "case_id": case_id,
        "all_models_boundary_valid": True,
        "hd95_improvement": hd95_gain,
        "boundary_dice_improvement": bd_gain,
    }


def test_boundary_ranking_prioritizes_same_case_improved_in_all_seeds():
    rows = []
    for seed in MODULE.SEEDS:
        rows.append(_boundary_row(seed, "consistent", 2.0, 0.05))
    rows.extend(
        [
            _boundary_row(42, "mixed", 20.0, 0.30),
            _boundary_row(55, "mixed", 20.0, 0.30),
            _boundary_row(123, "mixed", -1.0, -0.01),
        ]
    )

    ranked = MODULE.aggregate_boundary_cases(pd.DataFrame(rows))

    assert ranked.iloc[0]["case_id"] == "consistent"
    assert bool(ranked.iloc[0]["all_seeds_both_improved"])
    assert ranked.iloc[0]["both_improved_seeds"] == 3
    assert ranked.iloc[1]["both_improved_seeds"] == 2


def test_aggregation_rejects_missing_seed_for_an_item():
    incomplete = pd.DataFrame(
        [
            _small_row(42, "case", 0, 0.2, 0.3),
            _small_row(55, "case", 0, 0.2, 0.3),
        ]
    )

    try:
        MODULE.aggregate_small_lesions(incomplete)
    except ValueError as error:
        assert "all three seeds" in str(error)
    else:
        raise AssertionError("missing seed must be rejected")
