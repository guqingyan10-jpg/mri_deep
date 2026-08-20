# Gated Edge Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add zero-initialized adaptive Laplacian gating to the existing Edge and Edge+Boundary models, then expose a fair seed-123 paired training and four-metric evaluation workflow.

**Architecture:** Extend the shared `ResUpEdge` decoder with a `gated_concat` mode so both requested models reuse the same implementation. Parameterize the existing boundary subclass and training scripts, then add two gated jobs to the seed workflow and evaluation registry without importing the coworker's local patch-training stack.

**Tech Stack:** Python, PyTorch, argparse, pytest, existing ResUNet training/evaluation framework.

---

### Task 1: Add shared gated decoder fusion

**Files:**
- Modify: `models/resunet_edge.py`
- Create: `tests/test_gated_edge_integration.py`

- [ ] **Step 1: Write failing model contract tests**

Add tests that instantiate `ResUpEdge(..., fusion="gated_concat")`, assert the
gate weight and bias are zero, and compare concat/gated outputs after copying
all shared state tensors:

```python
def test_gated_concat_starts_as_exact_concat():
    concat = ResUpEdge(16, 4, 4, fusion="concat")
    gated = ResUpEdge(16, 4, 4, fusion="gated_concat")
    gated_state = gated.state_dict()
    for key, value in concat.state_dict().items():
        gated_state[key] = value
    gated.load_state_dict(gated_state)
    assert torch.count_nonzero(gated.edge_gate.weight) == 0
    assert torch.count_nonzero(gated.edge_gate.bias) == 0
    assert torch.allclose(concat(x1, x2, edge), gated(x1, x2, edge))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/test_gated_edge_integration.py -v`

Expected: failure because `gated_concat` is rejected and `edge_gate` is absent.

- [ ] **Step 3: Implement residual gated concat**

Extend accepted fusion values and create a 1x1 gate at each decoder block:

```python
if fusion in ("concat", "gated_concat"):
    total_in = in_channels + edge_channels
    self.conv = ResBlock(total_in, out_channels)
    if fusion == "gated_concat":
        self.edge_gate = nn.Conv3d(total_in, edge_channels, 1)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.zeros_(self.edge_gate.bias)
```

In `forward`, compute `edge_feat *= 1 + tanh(edge_gate(cat(...)))` before the
existing concat operation.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_gated_edge_integration.py -v`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add models/resunet_edge.py tests/test_gated_edge_integration.py
git commit -m "feat: add gated Laplacian edge fusion"
```

### Task 2: Reuse gated fusion in the boundary model and training CLIs

**Files:**
- Modify: `models/resunet_hf_concat_boundary.py`
- Modify: `scripts/train_v2_edge.py`
- Modify: `scripts/train_hf_concat_boundary.py`
- Modify: `tests/test_gated_edge_integration.py`
- Modify: `tests/test_hf_concat_boundary_contract.py`

- [ ] **Step 1: Write failing boundary and CLI tests**

Test that `ResUNetHFConcatBoundary(fusion="gated_concat")` is accepted, still
returns two tensors, and both training scripts expose `gated_concat`:

```python
model = ResUNetHFConcatBoundary(n_channels=8, fusion="gated_concat")
seg, boundary = model(torch.randn(1, 4, 16, 16, 16))
assert seg.shape == boundary.shape == (1, 3, 16, 16, 16)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/test_gated_edge_integration.py tests/test_hf_concat_boundary_contract.py -v`

Expected: failure because the boundary class and CLI choices do not accept the
new fusion value.

- [ ] **Step 3: Parameterize existing boundary and training paths**

Change the boundary constructor to:

```python
def __init__(self, in_channels=4, n_classes=3, n_channels=24,
             fusion="concat"):
    super().__init__(..., fusion=fusion, edge_type="laplacian")
```

Add `gated_concat` to `train_v2_edge.py --fusion` choices. Add `--fusion` with
choices `concat/gated_concat` to `train_hf_concat_boundary.py`, pass it to the
model, and include fusion in default gated checkpoint directory names. Leave
learning rate, epochs, accumulation, early stopping, loss, data, and warm-start
logic unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_gated_edge_integration.py tests/test_hf_concat_boundary_contract.py -v`

Expected: all focused tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add models/resunet_hf_concat_boundary.py scripts/train_v2_edge.py scripts/train_hf_concat_boundary.py tests
git commit -m "feat: expose gated edge boundary training"
```

### Task 3: Add the paired seed-123 gated runner

**Files:**
- Create: `scripts/run_gated_seed_screen.py`
- Modify: `tests/test_seed_stability_training.py`

- [ ] **Step 1: Write failing paired-job tests**

Test a `build_jobs(seed, output_root, epochs, lr)` function returning exactly:

```text
edge_laplacian_gated_concat
hf_gated_concat_boundary_w0.1
```

Assert both commands contain `--seed 123`, `--epochs 200`, `--lr 5e-4`, their
isolated checkpoint directory, and the exact same `--baseline_checkpoint`.

- [ ] **Step 2: Run seed tests and verify RED**

Run: `pytest tests/test_seed_stability_training.py -v`

Expected: failure because `run_gated_seed_screen.py` is absent.

- [ ] **Step 3: Implement the two-job runner**

Reuse `find_completed_baseline` from `run_seed_stability.py`. Default arguments:

```python
--seed 123
--output_root /root/autodl-tmp/stability
--epochs 200
--lr 5e-4
```

Run the Edge command first and Boundary command second with
`subprocess.run(..., check=True)`. Support `--dry_run` for command inspection.

- [ ] **Step 4: Verify runner tests and dry run**

Run:

```bash
pytest tests/test_seed_stability_training.py -v
python scripts/run_gated_seed_screen.py --seed 123 --dry_run
```

Expected: tests pass and two commands reference the same seed123 baseline best
checkpoint placeholder/path.

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/run_gated_seed_screen.py tests/test_seed_stability_training.py
git commit -m "feat: add paired gated seed screening runner"
```

### Task 4: Register gated models in four-indicator evaluation

**Files:**
- Modify: `scripts/eval_key_comparison.py`
- Modify: `tests/test_eval_key_comparison_seed.py`

- [ ] **Step 1: Write failing seed-registry tests**

Assert `build_seed_experiments(123, root)` includes the existing four models
plus:

```text
seed123/edge_laplacian_gated_concat
seed123/hf_gated_concat_boundary_w0.1
```

Assert their model kwargs contain `fusion="gated_concat"`, and assert `PRIMARY`
still contains exactly Macro Dice, ET Dice, Small-case ET Dice, and ET HD95.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `pytest tests/test_eval_key_comparison_seed.py -v`

Expected: failure because gated seed models are not registered.

- [ ] **Step 3: Add gated seed experiment entries**

Append the Edge gated spec using `ResUNetEdge` with Laplacian/gated kwargs and
the gated Boundary spec using `ResUNetHFConcatBoundary` with gated kwargs.
Preserve the existing four primary indicators and cached checkpoint behavior.

- [ ] **Step 4: Run evaluation and full regression verification**

Run:

```bash
pytest tests/test_eval_key_comparison_seed.py -v
pytest -q
python -m py_compile models/resunet_edge.py models/resunet_hf_concat_boundary.py scripts/train_v2_edge.py scripts/train_hf_concat_boundary.py scripts/run_gated_seed_screen.py scripts/eval_key_comparison.py
git diff --check
```

Expected: all tests pass, compilation succeeds, and diff check reports no
whitespace errors.

- [ ] **Step 5: Commit Task 4**

```bash
git add scripts/eval_key_comparison.py tests/test_eval_key_comparison_seed.py
git commit -m "feat: evaluate gated seed variants"
```

