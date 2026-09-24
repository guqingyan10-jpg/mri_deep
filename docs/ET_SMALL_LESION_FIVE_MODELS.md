# Five-model small and medium ET lesion evaluation

Run this on the AutoDL machine from the repository root after the five seed-55
best checkpoints and BraTS2020 images are available:

```bash
cd /root/autodl-tmp/mri_deep
python scripts/eval_et_small_lesion_five_models.py --dry-run
python scripts/eval_et_small_lesion_five_models.py
```

The script finds checkpoints in the original AutoDL directories (`Unet`,
`ResUNet_model`, `.autodl/attention-unet`, `nnUnet`, and
`ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model`). It also accepts packaged
seed-55 best checkpoints from the handoff bundle, if present. To override any
location, pass `--checkpoint MODEL=/absolute/path/to/best_model_*.pth` (or a
directory containing one). Model names are `unet`, `resunet`, `attention`,
`nnunet`, and `afbms`. Use `--csv /path/to/tumourCSV.csv` when the data manifest
is elsewhere. The script fails before inference if a best checkpoint is missing.

The fourth model is the project's **historical `nnUNet3d` style network** from
the original four-baseline comparison. It is not the newer nnU-Net v2
`PlainConvUNet` architecture.

The test protocol matches the formal ET lesion analysis: 37 fixed test cases,
probability threshold 0.33, 3D 26-connectivity, components of at least 10
voxels, a training-defined small stratum of 10–44 voxels, and a medium stratum
of 45–4678 voxels. There should be 31 small and 36 medium GT ET lesions.
`--strata-json /path/to/et_training_lesion_strata.json` optionally verifies
the frozen training thresholds. The script stops if the case count, lesion
counts, or GT lesion identities differ.

Outputs are written to `et_small_medium_lesion_five_models_results/`:

- `summary.csv`: one row per model for `small`, `medium`, and pooled
  `small_medium`, with matched Dice, GT-anchored Dice, recall, and counts.
- `small_medium_lesions.csv`: one row per GT small or medium lesion per model.
- `small_lesions.csv` / `medium_lesions.csv`: separate detail tables, with
  missed lesions scored as zero in `gt_anchored_dice`.
- `test_cases.csv`: the evaluated case IDs.

Matched lesion Dice averages successful one-to-one matches only. GT-anchored
lesion Dice averages across all GT lesions in the selected size group,
assigning zero to each unmatched lesion. The pooled row weights each GT lesion
equally across both strata. The existing formal seed-55 results provide a check: the
ResUNet values are 0.368461 matched and 0.035658 GT-anchored (3/31 detected),
and AFBMS-ResUNet values are 0.364706 and 0.023529 (2/31 detected). Small-case
ET Dice from the older overall evaluation is a different, case-level measure.
