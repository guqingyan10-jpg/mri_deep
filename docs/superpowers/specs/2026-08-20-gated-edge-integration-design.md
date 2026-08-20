# Gated Edge Integration Design

## Goal

Integrate the coworker's adaptive Laplacian gate into the existing project and
train one paired seed-123 screening run for two new variants:

1. Edge (Laplacian, gated concat) with `BCEDiceLoss`.
2. Edge (Laplacian, gated concat) with boundary supervision at weight `0.1`.

The integration must reuse the repository's existing architecture, loss,
training, checkpoint, data split, and evaluation logic. Local screening-only
changes from the coworker package are explicitly out of scope.

## Source Delta

The only unique model behavior to migrate is `gated_concat` in each
`ResUpEdge` decoder stage:

```text
gate_input = concat(skip, upsampled decoder, Laplacian feature)
scale = 1 + tanh(conv1x1(gate_input))
gated_edge = scale * Laplacian feature
output = ResBlock(concat(skip, upsampled decoder, gated_edge))
```

The gate convolution weight and bias are initialized to zero. Therefore the
initial scale is exactly one and the initial forward behavior matches the
existing concat model.

The following coworker-package changes will not be migrated:

- GT-centered patch sampling and screening-only datasets.
- `n_channels=8`, patch size 64, learning rate `1e-4`, or accumulation 2.
- MPS-specific compatibility changes.
- Duplicate boundary loss, evaluation, or trainer implementations already
  present in this repository.

## Architecture

### Shared Edge Backbone

Extend `ResUpEdge` and `ResUNetEdge` to accept `fusion="gated_concat"` while
preserving current `concat` and `add` behavior and checkpoint compatibility.
Each of the four decoder stages owns an independent zero-initialized gate.

The edge-only variant remains a single-output model and uses the existing
`out` segmentation head.

### Boundary Variant

Parameterize `ResUNetHFConcatBoundary` with a `fusion` argument whose default
remains `concat`. The gated boundary model uses the same multi-scale Laplacian
backbone and existing boundary head; only the decoder fusion changes to
`gated_concat`.

The model continues returning `(segmentation_logits, boundary_logits)` and
uses the existing `BCEDiceWithBoundaryLoss`.

## Training Interface

Reuse the two existing entrypoints:

- `train_v2_edge.py --fusion gated_concat --edge_type laplacian`
- `train_hf_concat_boundary.py --fusion gated_concat --boundary_weight 0.1`

Both scripts retain their current explicit `--seed`, `--checkpoint_dir`, and
`--baseline_checkpoint` controls. A small paired runner will create the two
seed-123 jobs under:

```text
/root/autodl-tmp/stability/seed123/edge_laplacian_gated_concat
/root/autodl-tmp/stability/seed123/hf_gated_concat_boundary_w0.1
```

Both jobs receive the exact best checkpoint from:

```text
/root/autodl-tmp/stability/seed123/baseline/best_model_*.pth
```

The runner must refuse to start if the baseline completion marker or best
checkpoint is missing.

## Fairness Contract

The new experiments use the repository's existing settings:

- Seed: 123 for the first screening run.
- Data CSV and split: `tumourCSV.csv`, `random_state=10`.
- Base channels: 24.
- Optimizer: Adam.
- Learning rate: `5e-4`.
- Scheduler: `ReduceLROnPlateau`, patience 2.
- Batch size: 1.
- Gradient accumulation: 4.
- Maximum epochs: 200.
- Early stopping: patience 25, `min_delta=1e-4`.
- Warm-start: the same seed-123 baseline best checkpoint.
- Checkpoint selection: minimum validation loss.

No pretrained Edge or Boundary checkpoint may be used to initialize either
new arm.

## Evaluation

Extend seed key comparison so seed 123 can include the two gated variants.
The reported primary indicators remain exactly:

- Macro Dice.
- ET Dice.
- ET HD95.
- Small-case ET Dice.

The principal comparisons are paired within seed 123:

- Existing Edge concat vs new Edge gated concat.
- Existing concat Boundary `w=0.1` vs new gated Boundary `w=0.1`.

## Compatibility And Failure Handling

- Existing `concat` and `add` model construction and forward behavior must not
  change.
- Existing HF concat boundary checkpoints remain loadable because its default
  fusion stays `concat`.
- Gated checkpoints contain new `edge_gate` tensors and must be loaded only
  into gated model configurations.
- Training scripts validate the supplied baseline checkpoint and require a
  `best_model_*.pth` name for fair warm-starting.
- Resume remains isolated to each experiment's own checkpoint directory.

## Verification

Tests will cover:

- `gated_concat` is accepted by model and training CLI contracts.
- Every gated decoder stage owns a zero-initialized gate.
- At initialization, gated concat produces the same output as concat when all
  shared weights are identical.
- The boundary model supports both concat and gated concat and still returns
  two outputs.
- The paired runner creates exactly two seed-isolated jobs and passes the same
  baseline checkpoint to both.
- Seed key evaluation registers both gated checkpoint directories and retains
  the four primary indicators.

