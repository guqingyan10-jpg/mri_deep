# Multi-Scale Context Paired Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add the colleague's bottleneck `MultiScaleContext3d` to the existing formal HF concat boundary pipeline and provide paired seed-42/123 AutoDL training and evaluation commands.

**Architecture:** Keep `ResUNetHFConcatBoundary` unchanged by default and enable the new residual multi-scale bottleneck only through an explicit flag. Reuse the existing trainer, seed-specific baseline completion checks, checkpoint policy, and key comparison evaluator.

**Tech Stack:** Python 3, PyTorch, pytest/unittest-compatible contract tests.

---

### Task 1: Define The Failing Integration Contract

**Files:**
- Create: `tests/test_multiscale_context_integration.py`

- [x] **Step 1: Add contract tests**

Test that the model source defines the four-branch context module, both model classes expose a disabled-by-default flag, the training entry point forwards the flag, the paired runner builds seed-specific fair commands, and seed evaluation registers the multiscale checkpoint.

- [x] **Step 2: Verify the test fails for the missing feature**

Run:

```powershell
py -3.11 -m unittest tests.test_multiscale_context_integration -v
```

Expected: failures for missing `MultiScaleContext3d`, runner, CLI flag, and evaluation registration.

### Task 2: Integrate The Optional Bottleneck Module

**Files:**
- Modify: `models/resunet_edge.py`
- Modify: `models/resunet_hf_concat_boundary.py`
- Modify: `scripts/train_hf_concat_boundary.py`

- [x] **Step 1: Add the module and opt-in flag**

Implement four parallel branches with dilation 1, 2, 3, and a 1x1 branch; concatenate, project, add the input residual, and apply ReLU. Instantiate and execute it only when `multiscale_context=True`.

- [x] **Step 2: Expose formal training configuration**

Add `--multiscale_context`, forward it to the model, print it in the run configuration, and use a distinct default checkpoint directory when enabled.

- [x] **Step 3: Run the focused contract test**

Run the Task 1 command. Model and CLI assertions should pass; runner and evaluation assertions remain red until Task 3.

### Task 3: Add Paired Training And Evaluation Registration

**Files:**
- Create: `scripts/run_multiscale_seed_screen.py`
- Modify: `scripts/eval_key_comparison.py`
- Modify: `tests/test_eval_key_comparison_seed.py`

- [x] **Step 1: Add the paired runner**

Build one sequential job per requested seed. Require the completed seed baseline and pass:

```text
--fusion concat --boundary_weight 0.1 --multiscale_context
--epochs 200 --lr 0.0005 --seed <seed>
```

Store outputs below `seed<seed>/hf_concat_boundary_w0.1_multiscale` and support `--dry_run`.

- [x] **Step 2: Register the model for seed evaluation**

Add the new directory with `model_kwargs={..., "multiscale_context": True}` to `build_seed_experiments`.

- [x] **Step 3: Run focused tests**

Run:

```powershell
py -3.11 -m unittest tests.test_multiscale_context_integration -v
py -3.11 -m unittest tests.test_seed_stability_training -v
```

Expected: all discovered tests pass.

### Task 4: Verify AutoDL Handoff

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document only the required commands**

Add dry-run, paired training, CUDA smoke test, and per-seed evaluation commands.

- [x] **Step 2: Verify syntax and command output**

Run:

```powershell
py -3.11 -m compileall models scripts tests
py -3.11 scripts/run_multiscale_seed_screen.py --seeds 42 123 --dry_run
```

The dry run requires actual AutoDL baseline completion markers, so local verification uses the runner command-construction test when those directories are absent.

- [x] **Step 3: Review the diff**

Run `git diff --check` and confirm no unrelated files changed.

