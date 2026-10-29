# Official nnU-Net v2: fixed-split BraTS2020 and UCSF-BMSR

This workflow calls the installed **nnU-Net v2 2.6.2** planner, preprocessor,
default trainer and predictor. The configuration is the standard
`nnUNetPlans/3d_fullres`, one fixed fold, one model, official mirror TTA and
`checkpoint_final.pth`. It is not an ensemble or a claim about the strongest
available nnU-Net preset. No project Adam/200-epoch trainer is substituted.

## Environment and repository

Run from the AutoDL Git checkout. Keep the original experiment environment intact.

```bash
git switch master
git pull --ff-only origin master
conda create -n nnunet_official python=3.10 -y
conda activate nnunet_official
mkdir -p /root/autodl-tmp/pip-tmp
export TMPDIR=/root/autodl-tmp/pip-tmp
python -m pip install --upgrade pip
python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-cache-dir -r requirements_nnunet_official.txt
python -c "import torch; from torch import GradScaler; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
```

The CUDA 12.4 wheel fits the previously reported NVIDIA 550 driver; another
machine needs a driver-compatible CUDA wheel. The pinned official trainer
requires modern `torch.GradScaler`, unavailable in the old PyTorch 2.1 environment.
Compilation is disabled for portability, without changing the official network,
optimizer, objective, augmentation or training budget. Actual versions are recorded.

## Fixed protocol

| Item | BraTS2020 | UCSF-BMSR |
| --- | --- | --- |
| Split source | Existing `tumourCSV.csv`, preserving row order and original split | Existing paired-study `manifest.csv`; no resplitting |
| Train / validation / test | 257 / 74 / 37 scans | 321 / 94 / 46 scans; 219 / 63 / 32 patients |
| Modalities | FLAIR, T1, T1ce, T2 | T1post, T1pre, FLAIR, subtraction |
| Native training input | Original full volume, original intensities and spacing | Original full volume and intensities, canonical RAS orientation |
| Labels | Only 4 becomes 3; WT=[1,2,3], TC=[1,3], ET=3 | Official binary background/metastasis task, two-channel softmax |
| Shared scoring GT | Original crop XYZ `[40:210,40:210,20:120]` | Exact labels and effective spacing from the existing NPZ cache |
| Prediction alignment | Native prediction cropped to the study FOV | Probabilities resized trilinearly before thresholding; integer masks resized with nearest-neighbor |

The default `--input-space native` lets nnU-Net perform its own normalization,
automatic planning, resampling and patch selection. Native UCSF GT is checked
against the previously used cached labels. Cached images are not used as native
training input. The official BraTS model receives context outside the study crop;
only scoring uses the same crop. Report these input/preprocessing and compute
differences: this is a system comparison, not a shared-training-recipe ablation.

`imagesTr/labelsTr` contain train+validation cases for official planning.
Test images are separate in `imagesTs`; test labels never enter planning or training.
`splits_final.json` has one train/val entry, explicitly used as fold 0.
The original seed-10 UCSF patient assignment is checked as well as its counts.
An explicitly different `--input-space study` experiment needs a new work directory.

## BraTS: prepare and train

The root below is the parent of `BraTS20_Training_047`. Replace it if necessary;
both `.nii` and `.nii.gz` are accepted. If CSV paths already work, omit the root.

```bash
python scripts/prepare_official_nnunet.py \
  --dataset brats \
  --brats-data-root /root/autodl-tmp/brats_project/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData \
  --work-dir /root/autodl-tmp/nnunet_official_brats

python -u scripts/run_official_nnunet.py \
  --experiment /root/autodl-tmp/nnunet_official_brats/experiment.json \
  --seed 55 --num-processes 2 --gpu-memory-target 8
```

## UCSF: use the original manifest

The example uses the original preparation script's default manifest location.
Substitute the existing Baseline/Full manifest if it was saved elsewhere.

```bash
python scripts/prepare_official_nnunet.py \
  --dataset ucsf \
  --ucsf-manifest /root/autodl-tmp/ucsf_bmsr_prepared/manifest.csv \
  --work-dir /root/autodl-tmp/nnunet_official_ucsf

python -u scripts/run_official_nnunet.py \
  --experiment /root/autodl-tmp/nnunet_official_ucsf/experiment.json \
  --seed 55 --num-processes 2 --gpu-memory-target 8
```

Native directories come from `source_dir`. If moved, pass `--ucsf-data-root`.
If the NPZ cache moved, pass `--ucsf-prepared-root` (its root or `cases` folder).
The two datasets train separately with isolated plans, checkpoints and outputs.

## Stages, resume and training cost

The runner performs `plan -> train -> predict -> evaluate`. Official training
retains 1000 epochs, 250 updates per epoch, SGD/poly LR, augmentation, foreground
oversampling, planned batch/patch sizes and deep supervision. BraTS uses region
Dice+BCE; UCSF uses the official categorical objective. There is no warm-start.

Use `--stage plan|train|predict|evaluate` for one stage, or `--dry-run` to inspect
commands. Interrupted training requires explicit `--resume` on the same command.
Resume restores official training state, not just weights. Progress since the last
periodic checkpoint may be recomputed. Completed stages are reused after checks.
The planning target is not a GPU allocation cap; select it before training and
retain it for that run. Worker count can be reduced to 1.

Default mirror TTA can be disabled with `--disable-tta` for a separately recorded
inference protocol. Do not overwrite results with a changed inference protocol.
GPU and multiworker training are not promised bitwise deterministic.

## Results to download

Outputs are under `<work-dir>/official_pipeline/seed55/`:

- `evaluation_native_default/`: official native integer decision, main system result;
- `evaluation_study033/`: aligned probabilities with the prespecified 0.33 threshold;
- `predictions/`: native NIfTI labels and probability arrays;
- `training_complete.json`, `training_resources.json`, `run_protocol.json` and logs;
- **`results_for_paper.zip`**: the small review archive, without MRI arrays or weights.

Download one archive per dataset. It contains plans, the frozen split, actual
budget, package/GPU details, aggregate metrics, per-case and per-lesion results.
Weights remain in `<work-dir>/nnUNet_results/seed55/<DatasetName>/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/`.

Evaluation reuses the project's 26-connected, minimum-10-voxel, one-to-one matcher.
BraTS small ET remains 10-44 voxels (31 test lesions), medium 45-4678 (36 lesions).
Misses count as zero for GT-anchored Dice; matched Dice includes detected GT only.
The manuscript profile uses physical pooled-distance HD95, explicit empty-mask
penalties and 1 mm surface Dice. GT and prediction checksums are saved.

Do not splice different metric definitions into one table. The evaluator accepts
existing study-space prediction exports with `--probability-layout study` and
`--model-name`. `--metric-profile legacy-ucsf` is a separate historical definition.
Neither analysis should be selected according to test performance. One seed is
not multi-seed evidence; report official compute separately and include AFBMS's
baseline pretraining cost. Separate dataset retraining is not zero-shot generalization.

Primary documentation: [custom splits](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md),
[region training](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md),
[official usage](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md).
