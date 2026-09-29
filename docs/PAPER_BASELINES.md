# DoubleBlock-ViT and SuperLightNet on the FULL BraTS split

These baselines use the authors' architecture files at fixed Git revisions:

| Model | Author code | Fixed commit | Configuration |
| --- | --- | --- | --- |
| DoubleBlock-ViT | [Laptq201/DoubleBlock-ViT-Unet-segment](https://github.com/Laptq201/DoubleBlock-ViT-Unet-segment) | `5b617d72608986f38e602f4dd69b95d45443f5c7` | `models/DB_MaxViT.py::Unet(4, 16, 3)`; authors report 7.8M parameters |
| SuperLightNet | [WTU-MIS-Laboratory/SuperLightNet](https://github.com/WTU-MIS-Laboratory/SuperLightNet) | `0d6532434586dd68750bf417b158511f46f4dd75` | `Jnetworks/superlightnet.py::NormalU_Net(4, 24, 4, depths_unidirectional="small")`; authors report 2.97M parameters |

DoubleBlock-ViT is a MaxViT-based Transformer hybrid. SuperLightNet's released implementation uses grouped multi-axis Hadamard product attention and convolution; it is best described as a lightweight attention model, not as a conventional self-attention Transformer baseline.

The [DoubleBlock-ViT paper](https://doi.org/10.1016/j.cmpb.2025.109165) and the [SuperLightNet CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_SuperLightNet_Lightweight_Parameter_Aggregation_Network_for_Multimodal_Brain_Tumor_Segmentation_CVPR_2025_paper.pdf) describe the architectures. The source files are downloaded on first run and checked against their Git revision and blob hash. They are not copied into this repository. DoubleBlock's two unguarded, import-time CUDA demo statements are skipped; the model definitions remain unchanged.

The SuperLightNet author code samples a viewing direction inside its forward pass even during evaluation. The evaluation script fixes the RNG seed immediately before its ordered test-case pass. Results from this network are therefore reproducible for the recorded environment and case order, but its single-pass inference remains stochastic by design.

The common environment pins MONAI 1.4.0, as in DoubleBlock's repository. SuperLightNet's README instead lists MONAI 1.3.0. The preflight is the compatibility check for the common environment; save the actual installed package versions with the paper results.

DoubleBlock outputs `[ET, TC, WT]`; the adapter reorders to this project's `[WT, TC, ET]`. SuperLightNet retains its original four-class head `[background, label 1, label 2, label 4]`. The authors train independent sigmoid logits for the three foreground labels; the adapter combines their probabilities with the Bernoulli OR rule to obtain nested `[WT, TC, ET]` region logits. The unused background channel remains in the original head. No convolution or attention block is changed. The adapter pads at the high end to dimensions the original networks accept and crops the prediction back to the project's dimensions. This is an architecture reproduction **under a common comparison protocol**, not a claim that the authors' paper Dice scores can be reproduced under their original training setups.

## Comparison protocol

The two new runs call `data.dataset.get_dataloader(BratsDataset, tumourCSV.csv, ...)` directly, as FULL does. This means the same BraTS 2020 CSV, fixed `random_state=10` train/validation/test split, same case order, four modality order (`flair, t1, t1ce, t2`), fixed `[40:210,40:210,20:120]` source-volume crop, per-modality min-max normalization, `[WT,TC,ET]` labels, and no augmentation. Default training uses the entire project crop; `--crop-size` is an explicit memory fallback that **changes the protocol**. The CSV's SHA-256 and all split IDs are recorded in `protocol.json`, and every referenced NIfTI file is checked before training.

The optimization settings mirror FULL: seed 55, Adam with default betas and no weight decay, learning rate `5e-4`, `ReduceLROnPlateau(mode="min", patience=2)`, batch size 1, gradient accumulation 4, up to 200 epochs, early stopping on validation loss after 25 epochs without an improvement of at least `1e-4`, and selecting `best_model_*.pth` on validation loss. The legacy FULL trainer carries an incomplete accumulation group into the next epoch; this script does the same and saves pending gradients for exact resume. Training never reads test images.

Two differences remain and must be stated with results. FULL's segmentation loss is `BCEDiceLoss` **plus 0.1 × boundary BCE** for its auxiliary boundary head; the paper baselines have no boundary head and train with the identical segmentation loss alone. Existing FULL weights were also **warm-started from a ResUNet best checkpoint**, while these two models initialize from their authors' code. For a comparison that controls initialization policy, retrain FULL from scratch in a separate directory using the command below. This does not remove the difference in the auxiliary boundary objective, which is part of FULL's proposed method.

## AutoDL commands

Use an AutoDL image with CUDA-enabled PyTorch 2.2 or newer and at least 24 GB VRAM (the SuperLightNet authors' minimum). Adjust the BraTS path in `tumourCSV.csv` only if your data is mounted elsewhere. An edited CSV changes its SHA-256; use the same edited file for every run and record it with the results.

```bash
cd /root/autodl-tmp
git clone https://github.com/guqingyan10-jpg/mri_deep.git
cd mri_deep
git fetch origin codex/paper-baselines
git switch -c codex/paper-baselines --track origin/codex/paper-baselines
python -m pip install -r requirements_paper_baselines.txt
python scripts/train_paper_baselines.py --model doubleblock --preflight
python scripts/train_paper_baselines.py --model superlight --preflight
```

The preflight verifies file paths, split, source revisions, parameter-count range and a small forward pass. Inspect its output before starting GPU training. Run each model independently; two simultaneous full-volume jobs can exceed GPU memory.

```bash
python -u scripts/train_paper_baselines.py --model doubleblock 2>&1 | tee /root/autodl-tmp/doubleblock_train.log
python -u scripts/train_paper_baselines.py --model superlight 2>&1 | tee /root/autodl-tmp/superlight_train.log
```

Interrupted run: rerun the same command with `--resume`. The two checkpoint directories are `/root/autodl-tmp/paper_baselines/doubleblock/seed_55` and `/root/autodl-tmp/paper_baselines/superlight/seed_55`; each contains `protocol.json`, the selected `best_model_*.pth`, `last_state.pth`, and `train_log.csv`. Never use `last_state.pth` for formal test reporting.

If full-volume training runs out of memory, use `--crop-size 128 128 96` **from the first epoch** in a new output directory. Validation and test then use 25%-overlap sliding-window inference. Crop training is not directly controlled against the existing full-volume FULL run; retrain FULL under the same crop policy before claiming a strictly matched training protocol.

After both trainings finish, evaluate on the frozen 37-case test split with the same 26-connected lesion matcher, threshold `0.33`, minimum component size 10 voxels, and training-derived ET size strata (`small=10–44`, `medium=45–4678`, `large>=4679`). GT-anchored Dice assigns zero to an unmatched GT lesion; matched Dice is conditional on a match.

```bash
python scripts/eval_paper_baselines.py \
  --full-checkpoint /root/autodl-tmp/ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model
```

Outputs are under `/root/autodl-tmp/paper_baselines/et_test_results/`. For a scratch-initialized FULL control run:

```bash
python -u scripts/train_hf_concat_boundary.py \
  --boundary_weight 0.1 --multiscale_context_v2 --seed 55 --from_scratch \
  --checkpoint_dir /root/autodl-tmp/FULL_scratch_seed_55
```

The paper baselines use different parameter budgets: DoubleBlock is near the 7.10M-parameter FULL network; SuperLightNet is about 2.97M. Report measured parameter counts from `protocol.json` alongside Dice. Keep the source revision, split hash, inference mode, and initialization policy in the methods section.
