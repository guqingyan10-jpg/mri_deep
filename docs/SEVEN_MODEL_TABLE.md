# Seven-model BraTS2020 comparison table

`scripts/build_seven_model_table.py` extends the supplied five-model table with
DoubleBlock-ViT, SuperLightNet, and a **matched small-lesion Dice** column. The
current reviewable template is [seven_model_comparison.md](seven_model_comparison/seven_model_comparison.md).
Blank cells mean the result has not been measured; they are never filled with
invented values. The five existing six-column rows retain the exact displayed
numbers from the supplied table.

Run from the project root **after training has reached its maximum epoch or
early stopping**. The script refuses to test a still-running or interrupted
paper-baseline training run. It checks `protocol.json`, the frozen test IDs,
the CSV hash and the author-source revision, and loads only the selected
`best_model_*.pth` checkpoint. It caches each finished test result against the
checkpoint SHA-256 and dataset identity, so rerunning skips unchanged models.

```bash
cd /root/autodl-tmp/mri_deep
python -u scripts/build_seven_model_table.py
```

Output is under `/root/autodl-tmp/paper_baselines/comparison_table/`:
`seven_model_comparison.md`, `seven_model_comparison.csv`, and per-model JSON
caches. The results should be copied back into the manuscript only after every
cell is filled and the seven checkpoint/protocol identities have been checked.

The old five models' Macro Dice, ET Dice, ET HD95, Boundary Dice, lesion F1 and
GT-anchored small-lesion Dice are **not rerun**. For their missing matched
small-lesion Dice, the script first looks for
`et_small_medium_lesion_five_models_results/summary.csv` and checks its
`test_cases.csv`, 37 test IDs, seed, 31 GT small lesions, and agreement with the
old table's rounded GT-anchored Dice. Supply `--legacy-summary /path/to/summary.csv`
if that prior evaluation is elsewhere. If the summary is absent, it reuses the
published seed-55 matched results for ResUNet and AFBMS-ResUNet, then loads
only the U-Net, Attention U-Net and nnU-Net-style checkpoints to compute their
missing matched small-lesion Dice. If a checkpoint is elsewhere, pass
`--checkpoint unet=/path/to/best_model_*.pth` (similarly `attention` or
`nnunet`). It stops if their GT-anchored Dice disagrees with the old table;
mixing different checkpoints or test cohorts would be misleading.

If only one new model has finished, run for that model while leaving the old
missing cells pending:

```bash
python -u scripts/build_seven_model_table.py --only-new --model superlight
python -u scripts/build_seven_model_table.py --only-new --model doubleblock
```

After both are finished, the default command fills the entire table using
valid old results and reuses the new-model caches. `--prepare` writes an
unmeasured table template without importing PyTorch or touching BraTS data.

The new models are evaluated **once per test case**, with one inference pass
providing all seven table metrics. Macro Dice is the mean of WT, TC and ET
case-mean Dice. ET HD95 excludes undefined empty-mask cases, as in the old
evaluator. The table's historical “Boundary Dice” values coincide with the
project's ET normalized surface Dice (NSD) at 1 mm tolerance; this is the
meaning retained for the new models. Lesion F1 is the **pooled/micro** ET F1
after 26-connected, minimum-10-voxel, one-to-one matching; it is **not** the
other evaluator's per-case mean lesion F1. The frozen small ET stratum is
10–44 GT voxels (31 GT lesions in the fixed 37-case test); matched Dice averages
only detected GT lesions while GT-anchored Dice includes zeros for misses.
The SuperLightNet test pass fixes random seeds and case order because the
released model samples a viewing direction even in evaluation mode.

The old table is a historical comparison. Its FULL row was originally trained
with ResUNet warm-start and a boundary auxiliary objective; the two paper
baselines start from scratch. Keep that distinction in the manuscript and
do not claim identical initialization for the historical FULL checkpoint.
