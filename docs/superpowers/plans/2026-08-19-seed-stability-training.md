# Seed Stability Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible multi-seed training for the BCEDice baseline, Laplacian-concat Edge model, and HF Concat Boundary variants.

**Architecture:** Every seed owns a separate checkpoint directory. The runner trains or resumes that seed's baseline first, locates its `best_model_*.pth`, then passes that exact path to the seed-matched Edge and HF experiments.

**Tech Stack:** Python, PyTorch, argparse, subprocess, pytest.

---

### Task 1: Define the stability runner

**Files:**
- Create: `scripts/run_seed_stability.py`
- Create: `tests/test_seed_stability_training.py`

- [ ] Write a failing unit test for a job builder that creates four isolated jobs per seed.
- [ ] Run `pytest tests/test_seed_stability_training.py -v` and confirm failure.
- [ ] Implement the runner with `--seeds`, `--output_root`, `--epochs`, `--lr`, and `--dry_run`.
- [ ] Re-run the targeted test and confirm success.

### Task 2: Add the baseline entrypoint

**Files:**
- Create: `scripts/train_baseline_bcedice.py`
- Modify: `tests/test_seed_stability_training.py`

- [ ] Test for `BCEDiceLoss`, `--seed`, and `--checkpoint_dir`.
- [ ] Confirm the test fails before implementation.
- [ ] Add a standalone ResUNet3d baseline that only resumes checkpoints in its supplied directory.
- [ ] Re-run the targeted test and confirm success.

### Task 3: Parameterize the derived-model scripts

**Files:**
- Modify: `scripts/train_v2_edge.py`
- Modify: `scripts/train_hf_concat_boundary.py`
- Modify: `tests/test_seed_stability_training.py`

- [ ] Test that both scripts expose `--seed`, `--checkpoint_dir`, and `--baseline_checkpoint`.
- [ ] Confirm the test fails before implementation.
- [ ] Preserve each default path, but use explicit seed, destination, and baseline path when supplied.
- [ ] Re-run the targeted test and confirm success.

### Task 4: Verify and deliver

**Files:**
- Test: `tests/test_seed_stability_training.py`
- Test: `tests/test_hf_concat_boundary_contract.py`
- Test: `tests/test_lesion_wise_matching.py`

- [ ] Run `python scripts/run_seed_stability.py --seeds 42 --dry_run` to inspect the four commands.
- [ ] Run `pytest -q`.
- [ ] Commit and push the scripts, tests, and plan to `origin/master`.
