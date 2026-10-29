# AFBMS-ResUNet

Research code for 3D ResUNet-based multimodal brain tumor MRI segmentation. The manuscript's main model combines four-level local high-frequency feature fusion (LHFC), auxiliary boundary supervision (ABS, loss weight **0.1**) during training, and adaptive residual multi-scale context (AR-MSC V2) at the bottleneck. The base channel count is 24. The BraTS2020 model predicts WT, TC, and ET; the model retrained on UCSF-BMSR predicts a single brain-metastasis mask. The earlier `w=0.3` experiments and V1/V3 context modules are not the manuscript's main model.

## Repository contents

| Path | Purpose |
| --- | --- |
| `models/`, `losses/`, `training/`, `data/` | Networks, losses, training, and data loading. The main model is defined in `models/resunet_hf_concat_boundary.py`. |
| `scripts/` | Training, evaluation, lesion analysis, and prediction-figure scripts for BraTS and UCSF. |
| `evaluation/` | Region-, boundary-, and lesion-level evaluation. |
| `tests/` | Tests for model connections, data splits, matching rules, and script contracts. |
| `docs/` | Focused instructions for retraining and comparison experiments. |
| `tumourCSV.csv` | BraTS2020 case list used as the source for this project's fixed split. |

Old notebooks, internal planning notes, and machine-specific dependency exports have been removed from the current tree. They remain available in Git history.

## Data and environment

The study splits **368 labeled BraTS2020 training cases** into 257 training, 74 validation, and 37 held-out test cases. The 37-case test set is this project's holdout, **not the official hidden BraTS test set**. See the [official CBICA page](https://www.med.upenn.edu/cbica/brats2020/data.html) for the original data and citation requirements; this project used a [Kaggle mirror](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data). The 461 publicly available labeled UCSF-BMSR scans were split by patient for retraining and evaluation on a separate dataset; see the [UCSF dataset page](https://imagingdatasets.ucsf.edu/dataset/1) for access and licensing. Do not commit raw MRI data or model weights to this repository.

Python 3.10 is recommended. Create an isolated environment, then run `pip install -r requirements.txt`. If your AutoDL environment already has a compatible CUDA build of PyTorch, you may retain it and install the remaining dependencies. The official nnU-Net v2 PlainConv architecture comparison additionally requires `requirements_nnunet_architecture.txt`. A UCSF-specific environment specification is retained in `requirements_ucsf.txt`.

The `path` column in `tumourCSV.csv` contains the original AutoDL data locations. On another machine, map these paths to the local BraTS image directory without changing case IDs, the random split, or test-set membership.

## Reproducing the main model

First train a ResUNet BCE–Dice baseline for a given seed. Then initialize the main model from that seed's best baseline checkpoint. For example, with seed 55 (replace the paths with your own):

```bash
python scripts/train_baseline_bcedice.py \
  --seed 55 --checkpoint_dir /path/to/seed55/baseline

python scripts/train_hf_concat_boundary.py \
  --seed 55 --fusion concat --boundary_weight 0.1 \
  --multiscale_context_v2 \
  --baseline_checkpoint /path/to/seed55/baseline/best_model_N.pth \
  --checkpoint_dir /path/to/seed55/full
```

To inspect the paired runs for seeds 42 and 123 before launching them:

```bash
python scripts/run_multiscale_v2_seed_screen.py --seeds 42 123 --dry_run
```

The main entry points for baseline comparison, ablation, and lesion-level evaluation are `scripts/eval_key_comparison.py`, `scripts/eval_et_lesion_stratified.py`, and `scripts/eval_et_small_lesion_five_models.py`. Use `scripts/eval_alpha_sensitivity.py` for the alpha-sensitivity analysis. Some scripts contain AutoDL default paths; check their arguments and checkpoint locations before running them elsewhere. See the [code review guide](CODE_REVIEW_GUIDE.md) for lesion matching and GT-anchored Dice checks.

The [UCSF instructions](docs/UCSF_EXTERNAL_BASELINE_FULL.md) cover patient-level splitting, Baseline/Full retraining, and consistent evaluation. This comparison involves **retraining within the UCSF dataset**, not zero-shot external testing. The scripts `scripts/select_boundary_typical_cases.py` and `scripts/run_multiseed_typical_case_analysis.py` generate representative figures from actual predictions.

## DoubleBlock-ViT architecture comparison

The authors' pinned DoubleBlock-ViT source and its project adapter are included for offline AutoDL training. See the [paper baseline instructions](docs/PAPER_BASELINES.md) for dependencies, the fixed BraTS split, preflight, training, and resume commands. Run `python scripts/train_paper_baselines.py --model doubleblock --preflight` before training. These are architecture comparisons under this project's training protocol; the source files do not establish any test-set performance claim.

## Official nnU-Net v2 system comparison

For the complete official nnU-Net v2 baseline on the frozen BraTS/UCSF cases,
see the [official system workflow](docs/OFFICIAL_NNUNET_SYSTEM.md). It uses native
MRI, official planning/preprocessing and the default 1000-epoch trainer, then
aligns predictions to the original study scoring grid. Run it in the documented
isolated environment; it is separate from the PlainConvUNet architecture-only run.

## Verification

After installing `pytest`, run `python -m pytest tests -q`. Model weights, raw MRI data, caches, manuscript drafts, and local results do not belong in the source repository and are excluded by `.gitignore`. Configure GPU, CUDA, and data paths for the AutoDL environment you use.
