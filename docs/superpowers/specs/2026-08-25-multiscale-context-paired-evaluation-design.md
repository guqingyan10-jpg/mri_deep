# Multi-Scale Context Paired Evaluation Design

## Objective

Determine whether the colleague's `MultiScaleContext3d` bottleneck module
improves the existing Laplacian EdgePyramid + concat + boundary-loss model
under this repository's established BraTS2020 split and full training
protocol.

The comparison covers seeds 42 and 123. The only experimental variable is
whether `MultiScaleContext3d` is enabled after the final encoder block.

## Compared Models

For each seed, compare these paired arms:

1. Existing control: `ResUNetHFConcatBoundary`, Laplacian EdgePyramid,
   decoder concat fusion, and `boundary_weight=0.1`.
2. New arm: the same model with `MultiScaleContext3d` applied to the
   bottleneck feature before decoder execution.

The module contains four parallel branches:

- 3x3x3 convolution with dilation 1;
- 3x3x3 convolution with dilation 2;
- 3x3x3 convolution with dilation 3;
- 1x1x1 convolution.

Their outputs are concatenated, projected back to the bottleneck channel
count, and added to the original feature through a residual connection.

## Fairness Contract

Both arms use the repository's existing formal training protocol:

- canonical `tumourCSV.csv`;
- deterministic split from `train_test_split(..., random_state=10)`;
- full `BratsDataset` preprocessing and 128x128x128 model input;
- seeds 42 and 123;
- 24 base channels;
- Adam optimizer;
- learning rate `5e-4`;
- batch size 1;
- gradient accumulation 4;
- at most 200 epochs;
- `ReduceLROnPlateau` with patience 2;
- early stopping with patience 25 and `min_delta=1e-4`;
- `BCEDiceWithBoundaryLoss(boundary_weight=0.1)`;
- the same best-checkpoint and last-checkpoint policies.

For each seed, the new arm must warm-start from that seed's completed
ResUNet baseline `best_model_*.pth`, exactly as the existing control arm did.
It must not warm-start from the control arm because that would change the
optimization path and invalidate the single-variable comparison.

The colleague package's GT-centered patch dataset, 8-channel model, five
epochs, and learning rate `1e-4` are excluded from formal training because
they do not match the current experiment protocol.

## Architecture Integration

`MultiScaleContext3d` will live with the edge backbone in
`models/resunet_edge.py`. `ResUNetEdge` will accept an optional
`multiscale_context` flag whose default is false. When disabled, its module
tree, state-dict keys, outputs, and initialization must remain compatible
with existing experiments.

`ResUNetHFConcatBoundary` will forward the same optional flag to its parent
and apply the context module between `enc4` and `dec1` when enabled. It will
continue returning `(segmentation_logits, boundary_logits)`.

No distance head, adaptive small-lesion loss, gated fusion, GT-centered
cropping, or unrelated colleague-package change is included.

## Training Orchestration

The existing boundary training entry point will expose the multiscale flag
instead of duplicating the optimizer or trainer logic. A dedicated runner
will create one new job per selected seed and will:

1. require the seed-specific baseline completion marker;
2. resolve the seed-specific baseline best checkpoint;
3. construct a command with concat fusion, boundary weight 0.1, the
   multiscale flag, 200 epochs, and learning rate `5e-4`;
4. use an isolated checkpoint directory;
5. execute seeds sequentially to avoid GPU contention;
6. support `--dry_run` so commands can be audited before training.

New checkpoint directories are:

```text
/root/autodl-tmp/stability/seed42/hf_concat_boundary_w0.1_multiscale
/root/autodl-tmp/stability/seed123/hf_concat_boundary_w0.1_multiscale
```

Interrupted training resumes only from the matching new experiment
directory. Missing or incomplete seed baselines stop execution with an
actionable error rather than silently selecting another checkpoint.

## Evaluation

The new models will be added to the existing seed-aware key comparison
workflow. Evaluation uses the same test cases, thresholding, cache policy,
and metric implementations as the existing control checkpoints.

The primary paired metrics are:

- Macro Dice, higher is better;
- ET Dice, higher is better;
- ET HD95, lower is better;
- Small-case ET Dice, higher is better.

Results must be reported per seed before any two-seed average. With only two
seeds, the aggregate is an early stability indication and not a final
statistical claim.

## Validation And Tests

Automated tests will cover:

- `MultiScaleContext3d` preserves the input tensor shape;
- the context-disabled model retains the existing forward contract;
- the context-enabled boundary model returns segmentation and boundary
  tensors at the input resolution;
- shared parameters have identical initialization between paired arms when
  the same seed is used;
- the new runner builds seed-specific commands with the required fairness
  arguments and seed-matched baseline checkpoints;
- evaluation registration points at the correct model configuration and
  checkpoint directories.

Before AutoDL training, local verification will include the focused test
suite, a small CPU forward/backward smoke test, and runner dry-run output for
seeds 42 and 123. Local verification does not attempt full training without
the AutoDL dataset and GPU checkpoints.

## Success Criteria

The adaptation is ready for AutoDL when:

- all focused and existing regression tests pass;
- the enabled module completes forward and backward execution;
- dry-run commands show identical training settings for seeds 42 and 123
  except for seed and checkpoint paths;
- existing checkpoints remain loadable when the context flag is disabled;
- the evaluation command includes both the control and multiscale variants
  for each requested seed.

The module is considered promising only after paired evaluation. A gain in
ET Dice or Small-case ET Dice must be considered together with Macro Dice and
ET HD95 rather than selecting a conclusion from one metric alone.
