# Architecture prediction assets

`generate_architecture_prediction_assets.py` exports real outputs for the full
AFBMS-ResUNet architecture figure. It uses the trained segmentation head and
auxiliary boundary head. The boundary mask is not made by eroding the predicted
segmentation, and a GT mask is never saved under a prediction filename.

Run from the repository root on AutoDL:

```bash
python scripts/generate_architecture_prediction_assets.py \
  --case-dir /root/autodl-tmp/MICCAI_BraTS2020_TrainingData/BraTS20_Training_309 \
  --checkpoint /root/autodl-tmp/mri_deep/deliverables/AFBMS_ResUNet_handoff_20260907/artifacts/weights/afbms_resunet/seed_55/best_model_epoch_090.pth \
  --out outputs/architecture_predictions
```

Replace both paths with the actual case directory and Full V2 checkpoint on the
server. The case directory must contain the five files
`CASE_flair.nii(.gz)`, `CASE_t1.nii(.gz)`, `CASE_t1ce.nii(.gz)`,
`CASE_t2.nii(.gz)`, and `CASE_seg.nii(.gz)`.

The command writes `seg_prediction_overlay.png`, `seg_prediction_mask.png`,
`boundary_head_mask.png`, `boundary_head_probability.png`, four MRI modality
slices, `hf_signed.png`, `prediction_arrays.npz`, and
`prediction_provenance.json`. It also creates a sibling
`architecture_predictions.zip` archive containing the same files. Send that
ZIP (or the complete output directory) back for insertion into the architecture
diagram. The JSON records the checkpoint hash, preprocessing, threshold,
selected slice, and output source.

For a binary model use `--n-classes 1`. The default three-channel mode follows
the project's BraTS order: WT, TC, ET. The script uses the exact crop and
normalization in `data/dataset.py` and loads the checkpoint with `strict=True`.
