# DoubleBlock-ViT and SuperLightNet on the FULL BraTS split

These baselines use the authors' architecture files at fixed Git revisions:

| Model | Author code | Fixed commit | Configuration |
| --- | --- | --- | --- |
| DoubleBlock-ViT | [Laptq201/DoubleBlock-ViT-Unet-segment](https://github.com/Laptq201/DoubleBlock-ViT-Unet-segment) | `5b617d72608986f38e602f4dd69b95d45443f5c7` | `models/DB_MaxViT.py::Unet(4, 16, 3)`; authors report 7.8M parameters |
| SuperLightNet | [WTU-MIS-Laboratory/SuperLightNet](https://github.com/WTU-MIS-Laboratory/SuperLightNet) | `0d6532434586dd68750bf417b158511f46f4dd75` | `Jnetworks/superlightnet.py::NormalU_Net(4, 24, 4, depths_unidirectional="small")`; authors report 2.97M parameters |

DoubleBlock-ViT is a MaxViT-based Transformer hybrid. SuperLightNet's released implementation uses grouped multi-axis Hadamard product attention and convolution; it is best described as a lightweight attention model, not as a conventional self-attention Transformer baseline.

The [DoubleBlock-ViT paper](https://doi.org/10.1016/j.cmpb.2025.109165) and the [SuperLightNet CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_SuperLightNet_Lightweight_Parameter_Aggregation_Network_for_Multimodal_Brain_Tumor_Segmentation_CVPR_2025_paper.pdf) describe the architectures. Exact author source files and their original licenses are bundled under `third_party/paper_baselines/` at the fixed Git revisions above. The loader verifies their SHA-256 before constructing a model. It makes **no network request** during training, so the AutoDL container only needs to pull this repository. DoubleBlock's two unguarded, import-time CUDA demo statements are skipped; the model definitions remain unchanged. DoubleBlock's source is MIT-licensed and SuperLightNet's is GPL-3.0-licensed; their original license files are included beside the source.

SuperLightNet's published `Jtrain.py` imports `Jnetworksv2.JCMNetv8`, a module missing from the pinned public checkout. This adapter instantiates the released `Jnetworks/superlightnet.py::NormalU_Net` directly. It reproduces that public architecture file, but an exact match to the authors' unpublished training module or paper checkpoint cannot be verified from the available source.

The SuperLightNet author code samples a viewing direction inside its forward pass even during evaluation. The evaluation script fixes the RNG seed immediately before its ordered test-case pass. Results from this network are therefore reproducible for the recorded environment and case order, but its single-pass inference remains stochastic by design.

The common environment pins MONAI 1.4.0, as in DoubleBlock's repository. SuperLightNet's README instead lists MONAI 1.3.0. The preflight is the compatibility check for the common environment; save the actual installed package versions with the paper results.

DoubleBlock outputs `[ET, TC, WT]`; the adapter reorders to this project's `[WT, TC, ET]`. SuperLightNet retains its original four-class head `[background, label 1, label 2, label 4]`. The authors train independent sigmoid logits for the three foreground labels; the adapter combines their probabilities with the Bernoulli OR rule to obtain nested `[WT, TC, ET]` region logits. The unused background channel remains in the original head. No convolution or attention block is changed. The adapter pads at the high end to dimensions the original networks accept and crops the prediction back to the project's dimensions. This is an architecture reproduction **under a common comparison protocol**, not a claim that the authors' paper Dice scores can be reproduced under their original training setups.

## What differs from the published implementations

| Item | Authors' released setup | This comparison | Architecture changed? |
| --- | --- | --- | --- |
| DoubleBlock-ViT body | `Unet(in_channels=4, n_channels=16, n_classes=3)`, MaxViT-based encoder, dual-path skips, project-and-excite | The **same pinned class and channel configuration**, initialized from scratch | No. Two import-time CUDA demo statements are skipped. |
| DoubleBlock-ViT output | Three logits in `[ET,TC,WT]` order | Reordered to the project's `[WT,TC,ET]` | No learned layers changed. |
| SuperLightNet body | Public `NormalU_Net(4,24,4,depths_unidirectional="small")`, Hadamard-product attention | The **same pinned public class and configuration**, initialized from scratch | No changes to released network layers. The authors' separate training entrypoint references an unavailable module, so identity with that missing version cannot be proved. |
| SuperLightNet output | Four logits for background and BraTS labels 1, 2, 4; positive classes trained with independent sigmoids | Original four-channel head retained; foreground probabilities combined into nested WT/TC/ET logits | No learned layers changed; the target/output adapter changes the optimization objective. |
| Spatial compatibility | Authors use their own fixed-size patches/preprocessing | High-end zero padding to model-compatible sizes, then crop back to the project tensor shape | No learned layers changed; boundary context differs from paper inputs. |
| Data and optimizer | DoubleBlock reports BraTS 2020/2021, 128³ crops, augmentation, AdamW `3e-4`; SuperLightNet reports BraTS 2019/2021 and an AdamW `1e-3` training example | Both use this project's fixed BraTS 2020 cases, no augmentation, Adam `5e-4` | Training protocol deliberately changed for comparison with FULL. Paper Dice scores are not directly comparable. |

The previous AutoDL script attempted to clone the author repositories at runtime and could fail when the container could not reach GitHub. Once this branch is updated, a partial checkout left at `/root/autodl-tmp/paper_baseline_sources/` is harmless: bundled sources take precedence. Do not retry the author-repository clones.

Both new models are **parallel baselines**, like the existing ResUNet and nnU-Net runs. A new run loads neither a ResUNet checkpoint nor an author checkpoint. `--resume` restores only that *same model's* interrupted run, with its optimizer, scheduler and RNG state.

## Comparison protocol

The two new runs call `data.dataset.get_dataloader(BratsDataset, tumourCSV.csv, ...)` directly, as FULL does. This means the same BraTS 2020 CSV, fixed `random_state=10` train/validation/test split, same case order, four modality order (`flair, t1, t1ce, t2`), fixed `[40:210,40:210,20:120]` source-volume crop, per-modality min-max normalization, `[WT,TC,ET]` labels, and no augmentation. Default training uses the entire project crop; `--crop-size` is an explicit memory fallback that **changes the protocol**. The CSV's SHA-256 and all split IDs are recorded in `protocol.json`, and every referenced NIfTI file is checked before training.

The optimization settings mirror FULL: seed 55, Adam with default betas and no weight decay, learning rate `5e-4`, `ReduceLROnPlateau(mode="min", patience=2)`, batch size 1, gradient accumulation 4, up to 200 epochs, early stopping on validation loss after 25 epochs without an improvement of at least `1e-4`, and selecting `best_model_*.pth` on validation loss. The legacy FULL trainer carries an incomplete accumulation group into the next epoch; this script does the same and saves pending gradients for exact resume. Training never reads test images.

The project tensor after axis reordering is `[4,100,170,170]`; labels are `[3,100,170,170]`. Preflight reads only the first training case to verify those shapes and finite values, then runs a small synthetic forward pass. The full training set is not scanned during preflight.

| Fairness item | FULL, with `--from_scratch` | DoubleBlock-ViT | SuperLightNet | Assessment |
| --- | --- | --- | --- | --- |
| BraTS cases and split | `tumourCSV.csv`, `random_state=10` | Same project loader | Same project loader | Matched; CSV SHA-256 and case IDs saved. |
| Preprocessing and targets | Four MRI modalities, fixed source crop, min-max, WT/TC/ET | Same loader and labels | Same loader and labels via output adapter | Matched input/target protocol; model-specific high-end padding is necessary. |
| Initialization | Random from scratch **only if** `--from_scratch` is passed | Random author-model initialization, no checkpoint | Random author-model initialization, no checkpoint | Matched policy for a new FULL scratch run. Historical FULL is **not** matched: it warm-started from ResUNet. |
| Learning setup | Seed 55, Adam `5e-4`, batch 1, accumulation 4, plateau patience 2, max 200 epochs | Same | Same | Matched settings and case order. |
| Checkpoint rule | Lowest validation objective; early stopping patience 25, `min_delta=1e-4` | Same rule | Same rule | Rule matched; objectives differ because FULL has an auxiliary term. |
| Segmentation objective | `BCEDiceLoss(WT,TC,ET)` + `0.1 × boundary BCE` | `BCEDiceLoss(WT,TC,ET)` | `BCEDiceLoss(WT,TC,ET)` | Same main segmentation loss; FULL's boundary supervision is a method-specific difference. |
| Parameter budget | 7,102,735 trainable parameters | Authors report ~7.8M; script logs exact count | Authors report ~2.97M; script logs exact count | Report actual counts alongside results; SuperLightNet has a smaller budget. |
| ET lesion test | 37 fixed test cases, threshold 0.33, 26-connectivity, min 10 voxels | Same evaluation function | Same function, RNG fixed before one pass | Matched scoring; SuperLightNet's public forward remains stochastic. |

Do **not** describe a comparison to the historical warm-started FULL checkpoint as having matched initialization. Retrain FULL from scratch in a fresh checkpoint directory for that claim. The boundary term and different validation objectives remain part of the proposed FULL method and must be stated with results.

## AutoDL commands

Use an AutoDL image with CUDA-enabled PyTorch 2.2 or newer. The SuperLightNet authors list 24 GB VRAM for their original setup; this project's padded full volumes can need more, especially for DoubleBlock. Prefer a larger GPU for the default full-volume protocol. Adjust the BraTS path in `tumourCSV.csv` only if your data is mounted elsewhere. An edited CSV changes its SHA-256; use the same edited file for every run and record it with the results.

```bash
cd /root/autodl-tmp
git clone https://github.com/guqingyan10-jpg/mri_deep.git
cd mri_deep
git fetch origin
git switch -c codex/paper-baselines --track origin/codex/paper-baselines
python -m pip install -r requirements_paper_baselines.txt
python scripts/train_paper_baselines.py --model doubleblock --preflight
python scripts/train_paper_baselines.py --model superlight --preflight
```

The preflight verifies file paths, split, source revisions, parameter-count range and a small forward pass. Inspect its output before starting GPU training. Run each model independently; two simultaneous full-volume jobs can exceed GPU memory.

If this branch is already checked out on AutoDL, update it with `git pull --ff-only` instead of running the first-time `git switch -c` command. The `torchvision.io` image-extension warning does not block these 3D MRI models; a GitHub clone timeout was the earlier training blocker.

```bash
python -u scripts/train_paper_baselines.py --model doubleblock 2>&1 | tee /root/autodl-tmp/doubleblock_train.log
python -u scripts/train_paper_baselines.py --model superlight 2>&1 | tee /root/autodl-tmp/superlight_train.log
```

Interrupted run: rerun the same command with `--resume`. The two checkpoint directories are `/root/autodl-tmp/paper_baselines/doubleblock/seed_55` and `/root/autodl-tmp/paper_baselines/superlight/seed_55`; each contains `protocol.json`, the selected `best_model_*.pth`, `last_state.pth`, and `train_log.csv`. Never use `last_state.pth` for formal test reporting.

If full-volume training runs out of memory, use `--crop-size 96 128 128` **from the first epoch** in a new output directory. The first dimension must not exceed the project's depth of 100. Validation and test then use 25%-overlap sliding-window inference. Crop training is not directly controlled against the existing full-volume FULL run; retrain FULL under the same crop policy before claiming a strictly matched training protocol.

After both trainings finish, evaluate on the frozen 37-case test split with the same 26-connected lesion matcher, threshold `0.33`, minimum component size 10 voxels, and training-derived ET size strata (`small=10–44`, `medium=45–4678`, `large>=4679`). GT-anchored Dice assigns zero to an unmatched GT lesion; matched Dice is conditional on a match. For matched initialization, train FULL from scratch first:

```bash
python -u scripts/train_hf_concat_boundary.py \
  --boundary_weight 0.1 --multiscale_context_v2 --seed 55 --from_scratch \
  --checkpoint_dir /root/autodl-tmp/FULL_scratch_seed_55
python scripts/eval_paper_baselines.py \
  --full-checkpoint /root/autodl-tmp/FULL_scratch_seed_55
```

Outputs are under `/root/autodl-tmp/paper_baselines/et_test_results/`. The scratch FULL checkpoint directory must be new and empty before starting; the historical FULL run is not a substitute for the matched-initialization comparison.

The paper baselines use different parameter budgets: DoubleBlock is near the 7.10M-parameter FULL network; SuperLightNet is about 2.97M. Report measured parameter counts from `protocol.json` alongside Dice. Keep the source revision, split hash, inference mode, and initialization policy in the methods section.
