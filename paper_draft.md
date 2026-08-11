# Enhancing Brain Tumor Segmentation via Multi-Scale Boundary and Lesion-Aware Learning

> **Authors:** [Your Name], [Co-authors]
> **Affiliation:** [Your Institution]
> **Conference:** MICCAI 2025 / Medical Image Analysis
> **Status:** Draft — metrics to be filled after running `eval_comprehensive.py`

---

## Abstract

**Background:** Accurate segmentation of brain tumors in multi-modal MRI is critical for diagnosis, treatment planning, and monitoring. While 3D segmentation models achieve high overall Dice scores on the BraTS benchmark, performance on small and multi-focal enhancing tumor (ET) regions remains systematically poor — 42% of ET lesions are smaller than 50 voxels and invisible to volume-dominant metrics.

**Methods:** We systematically investigate three orthogonal enhancement dimensions on a ResUNet3d baseline: (1) loss function engineering with boundary-aware and lesion-level objectives, (2) architectural modifications with explicit edge and frequency-enhanced branches, and (3) data-level foreground-aware patch sampling. In total, 15 models spanning these dimensions are evaluated against a common baseline.

**Results:** [TO BE FILLED — see tables below]

**Conclusion:** [TO BE FILLED — summarize best-performing approach and key insights]

**Keywords:** Brain tumor segmentation, BraTS, boundary loss, lesion-wise metrics, multi-scale learning, frequency enhancement

---

## 1. Introduction

Brain tumor segmentation from multi-parametric MRI (T1, T1ce, T2, FLAIR) is a fundamental task in medical image analysis. The BraTS challenge has driven substantial progress, with state-of-the-art methods achieving Dice scores above 0.85 for whole tumor (WT) and above 0.75 for enhancing tumor (ET) [1,2]. However, these aggregate metrics mask a critical failure mode: **small and multi-focal enhancing tumor lesions are systematically missed.**

Our analysis of the BraTS2020 training set (368 cases) reveals:

| Finding | Value |
|---|---|
| ET lesions smaller than 50 voxels | **42.0%** (412/981) |
| Multi-focal ET cases (≥2 lesions) | **50.3%** (185/368) |
| "Main + satellite fragments" pattern | **75.5%** of cases |
| Median ET lesion volume | **113 voxels** (~0.11 cm³) |
| ET volume as % of whole tumor | **~0.5%** (severe class imbalance) |

These statistics motivate a fundamental rethinking of both the training objectives and evaluation protocols. In this work, we systematically explore three dimensions of enhancement:

1. **Loss Functions (V1):** Can boundary-aware losses improve edge quality without sacrificing volume segmentation?
2. **Architecture (V2):** Do explicit edge extraction branches or frequency-domain feature enhancement improve fine-structure delineation?
3. **Data Sampling (SLA-FB):** Can foreground-aware patch sampling and lesion-level loss functions address the class imbalance at its root?

We evaluate all approaches on a comprehensive 20-metric panel spanning pixel-level classification, per-class segmentation, boundary quality, lesion-level detection, and computational efficiency.

---

## 2. Methods

### 2.1 Baseline Architecture

**ResUNet3d** serves as the common baseline. The architecture uses:

| Component | Specification |
|---|---|
| Encoder blocks | ResBlock (Conv3d → GroupNorm → ReLU) × 2 + residual |
| Downsampling | MaxPool3d (kernel=2) |
| Decoder blocks | ResBlock + Upsample(trilinear) + skip connection |
| Output | Conv3d(24 → 3) with Sigmoid |
| Normalization | GroupNorm (num_groups=8) |
| Activation | ReLU |
| Parameters | 5,763,867 (n_channels=24) |
| Loss | BCEDiceLoss = BCE + Dice |

**Training configuration (all experiments):** Adam optimizer (lr=5e-4), ReduceLROnPlateau (patience=2), batch_size=1, gradient accumulation=4, early stopping (patience=25, min_delta=1e-4). Data split: `train_test_split(random_state=10, test_size=0.3)`. Preprocessing: center crop `[40:210, 40:210, 20:120]` + min-max normalization.

### 2.2 V1: Loss Function Engineering

We replace BCEDiceLoss with a three-component loss:

$$L = \alpha \cdot L_{Dice} + \beta \cdot L_{CE} + \lambda_b \cdot L_{Boundary}$$

where $L_{Boundary}$ is the Kervadec et al. [3] distance-transform-based boundary loss, and class weights are WT=1.0, TC=3.0, ET=5.0. We fix α=1.0, β=0.5 and ablate λb ∈ {0.1, 0.3, 0.5}.

| Experiment | λb | Loss Formula | Modification from Baseline |
|---|---|---|---|
| **Baseline** | — | BCE + Dice | — |
| **λb=0.1** | 0.1 | Dice + CE + 0.1·BD | Loss function only |
| **λb=0.3** | 0.3 | Dice + CE + 0.3·BD | Stronger boundary weight |
| **λb=0.5** | 0.5 | Dice + CE + 0.5·BD | Maximum boundary weight |

*Warm-start:* All V1 models initialized from baseline checkpoint (epoch 199) for convergence stability.

### 2.3 V2: Architectural Modifications

#### 2.3.1 Edge Branch (Sobel / Laplacian)

We add a parallel edge detection branch to ResUNet3d. The edge extractor operates on raw 4-channel MRI input, producing a single-channel edge map via:

- **SobelEdge3d:** 3D Sobel operator (1st derivative), applied per-channel then fused via 1×1×1 conv
- **LaplacianEdge3d:** $I - \text{GaussianBlur}(I)$ (2nd derivative), capturing zero-crossings

The edge features are fused into each decoder level via either **concatenation** or **addition**.

| Experiment | Edge Type | Fusion | Parameters | Modification |
|---|---|---|---|---|
| **Baseline** | — | — | 5,763,867 | — |
| **Edge (Sobel, concat)** | Sobel (1st deriv.) | Concat | +300K | Edge branch added |
| **Edge (Sobel, add)** | Sobel (1st deriv.) | Add | +300K | Fusion strategy changed |
| **Edge (Laplacian, concat)** | Laplacian (2nd deriv.) | Concat | +300K | Edge operator changed |

#### 2.3.2 FGFE: Feature-level Frequency Enhancement

Inspired by BraTS-UMamba (Yao et al., MICCAI 2025) [4], FGFE operates on **decoder feature maps** rather than raw MRI:

1. **Laplacian Pyramid Decomposition:** $F \rightarrow F_h$ (high-freq) + $F_l$ (low-freq)
2. **Cross-Attention:** $F_h$ and $F_l$ independently query the skip-connection features $F_s$
3. **Residual Fusion:** $F_{out} = F_s + \text{Conv}([F_h^{attn}, F_l^{attn}])$

Unlike the Edge Branch (data-level, fixed operator), FGFE is **learnable** and operates at the **feature level**.

#### 2.3.3 HF Boundary Branch

Combines explicit edge extraction with a dedicated boundary prediction decoder:

1. Laplacian edge extraction from raw MRI
2. Parallel boundary decoder path → boundary prediction map
3. Boundary output supervised with BD loss (weight w ∈ {0.2, 0.3})
4. Attention-gated fusion with main decoder output

| Experiment | Edge Source | BD Loss Weight | Parameters | Modification |
|---|---|---|---|---|
| **HF Boundary** | Laplacian | 0.2 | +500K | Boundary decoder + attention gate |
| **HF Boundary+** | Laplacian | 0.3 | +500K | Stronger boundary supervision |

### 2.4 SLA-FB: Small Lesion-Aware Foreground-Background Learning

#### 2.4.1 Foreground-Aware Patch Sampling

Inspired by STSNet (Zhao et al., 2025) [5], we implement a 4-strategy weighted 3D patch sampler:

| Strategy | Probability | Description |
|---|---|---|
| `random` | 20% | Uniform random crop (standard) |
| `foreground` | 30% | Center at any WT tumor pixel |
| `et_centered` | 30% | Center at ET connected-component centroid (equal weight per lesion) |
| `small_lesion` | 20% | Center at small ET lesion (<50 vox) centroid only |

Key design decisions: (1) online real-time sampling (no disk augmentation), (2) "high-frequency exposure" replaces physical lesion magnification, (3) fallback chain: small_lesion → et_centered → foreground → random when pool is empty. Model architecture and loss unchanged — single-variable experiment.

#### 2.4.2 Lesion-Level Loss Functions

We implement three loss functions targeting the "Dice dominated by large lesions" problem:

| Loss | Formula | Level | Key Property |
|---|---|---|---|
| **CC-Dice** | $\frac{1}{K}\sum_{k=1}^{K}\text{Dice}(P_k, T_k)$ | Connected component | Each ET lesion equal weight |
| **PM-Dice (γ=2)** | $\text{Dice}(|y-\hat{p}|^\gamma \odot y, |y-\hat{p}|^\gamma \odot \hat{p})$ | Pixel | Upweights hard/misclassified pixels |
| **Full Combined** | Global Dice + CC-Dice + PM-Dice | Multi-scale | Volume + component + pixel |

where $K$ is the number of ET connected components per case. We further ablate whether global Dice is needed:

| Experiment | Global Dice | CC-Dice | PM-Dice | Modification |
|---|---|---|---|---|
| **CC-Dice** | ✅ | ✅ | — | Component-level Dice added |
| **PM-Dice (γ=2)** | ✅ | — | ✅ | Power-mean modulation |
| **BCE+CC-Dice** | — | ✅ | — | Ablation: no global Dice |
| **BCE+PM-Dice** | — | — | ✅ | Ablation: no global Dice |
| **Full Combined** | ✅ | ✅ | ✅ | All three combined |

---

## 3. Experimental Setup

### 3.1 Dataset

BraTS2020 Training Set: 368 cases (292 HGG, 76 LGG), 4 modalities (T1, T1ce, T2, FLAIR), 3 tumor sub-regions (WT, TC, ET). Ground truth annotations follow the standard BraTS protocol.

### 3.2 Data Split

70% training / 30% testing via `train_test_split(random_state=10, test_size=0.3)`. All models use identical data split (`tumourCSV.csv`).

### 3.3 Evaluation Protocol

All models evaluated at best validation-loss checkpoint with binarization threshold 0.33. **20 metrics across 5 categories:**

| Category | Metrics | Count |
|---|---|---|
| **Pixel Classification** | Accuracy, Precision, Recall, F1 (per WT/TC/ET) | 12 |
| **Segmentation** | Dice, Jaccard/IoU (per WT/TC/ET) | 6 |
| **Boundary Quality** | HD95 (ET, TC), NSD τ=1mm (ET, TC) | 4 |
| **Lesion Detection** | Lesion-wise Recall, Lesion-wise Precision, Overall Lesion Recall | 3 |
| **Small Lesion** | Small-case ET Dice (bottom 25% ET volume) | 1 |
| **Efficiency** | Parameters, Inference time | 2 |

*Note: HD95 = 95th percentile Hausdorff Distance (mm, lower is better); NSD = Normalized Surface Dice at τ=1mm margin (higher is better).*

---

## 4. Results

> **Instructions:** Run `python scripts/eval_comprehensive.py` on the server, then fill in the tables below from `comprehensive_results/comprehensive_results.csv`.

### 4.1 Per-Class Segmentation (Dice ± std)

| Model | WT Dice | TC Dice | ET Dice | WT Jaccard | TC Jaccard | ET Jaccard |
|---|---|---|---|---|---|---|
| Baseline (BCEDice) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| λb=0.1 (Dice+CE+0.1·BD) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| λb=0.3 (Dice+CE+0.3·BD) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| λb=0.5 (Dice+CE+0.5·BD) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| Edge (Sobel, concat) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| Edge (Sobel, add) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| Edge (Laplacian, concat) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| FGFE (Freq. Enhancement) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| HF Boundary (Laplacian, w=0.2) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| HF Boundary+ (Laplacian, w=0.3) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| FG Sampling (4-strategy) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| CC-Dice Loss | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| PM-Dice Loss (γ=2) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| BCE+CC-Dice (no Global) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| BCE+PM-Dice (no Global) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |
| Full (Global+CC+PM) | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ | ______ ± ______ |

### 4.2 Per-Pixel Classification (ET class)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Baseline (BCEDice) | ______ | ______ | ______ | ______ |
| λb=0.1 | ______ | ______ | ______ | ______ |
| λb=0.3 | ______ | ______ | ______ | ______ |
| λb=0.5 | ______ | ______ | ______ | ______ |
| Edge (Sobel, concat) | ______ | ______ | ______ | ______ |
| Edge (Sobel, add) | ______ | ______ | ______ | ______ |
| Edge (Laplacian, concat) | ______ | ______ | ______ | ______ |
| FGFE | ______ | ______ | ______ | ______ |
| HF Boundary | ______ | ______ | ______ | ______ |
| HF Boundary+ | ______ | ______ | ______ | ______ |
| FG Sampling | ______ | ______ | ______ | ______ |
| CC-Dice Loss | ______ | ______ | ______ | ______ |
| PM-Dice Loss (γ=2) | ______ | ______ | ______ | ______ |
| BCE+CC-Dice | ______ | ______ | ______ | ______ |
| BCE+PM-Dice | ______ | ______ | ______ | ______ |
| Full Combined | ______ | ______ | ______ | ______ |

*[Complete WT and TC pixel classification tables in `comprehensive_results/paper_table_comprehensive.md`]*

### 4.3 Boundary Quality & Lesion Detection

| Model | ET HD95↓ (mm) | ET NSD↑ | TC HD95↓ (mm) | TC NSD↑ | Lesion Recall↑ | Lesion Prec.↑ | Small ET Dice↑ |
|---|---|---|---|---|---|---|---|
| Baseline | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| λb=0.1 | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| λb=0.3 | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| λb=0.5 | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| Edge (Sobel, concat) | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| Edge (Sobel, add) | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| Edge (Laplacian, concat) | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| FGFE | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| HF Boundary | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| HF Boundary+ | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| FG Sampling | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| CC-Dice Loss | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| PM-Dice Loss (γ=2) | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| BCE+CC-Dice | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| BCE+PM-Dice | ______ | ______ | ______ | ______ | ______ | ______ | ______ |
| Full Combined | ______ | ______ | ______ | ______ | ______ | ______ | ______ |

*↓ = lower is better; ↑ = higher is better. Best in each column in **bold**.*

### 4.4 Training Efficiency

| Model | Epochs | Best Epoch | Train Time/Epoch (s) | Total Train Time (h) | #Params | Inference (s/case) |
|---|---|---|---|---|---|---|
| Baseline | ______ | ______ | ______ | ______ | 5,763,867 | ______ |
| λb=0.1 | ______ | ______ | ______ | ______ | 5,763,867 | ______ |
| λb=0.3 | ______ | ______ | ______ | ______ | 5,763,867 | ______ |
| λb=0.5 | ______ | ______ | ______ | ______ | 5,763,867 | ______ |
| Edge (Sobel, concat) | ______ | ______ | ______ | ______ | ~6,064K | ______ |
| Edge (Sobel, add) | ______ | ______ | ______ | ______ | ~6,064K | ______ |
| Edge (Laplacian, concat) | ______ | ______ | ______ | ______ | ~6,064K | ______ |
| FGFE | ______ | ______ | ______ | ______ | ~6,264K | ______ |
| HF Boundary | ______ | ______ | ______ | ______ | ~6,264K | ______ |
| HF Boundary+ | ______ | ______ | ______ | ______ | ~6,264K | ______ |
| FG Sampling | ______ | ______ | ______ | ______ | 5,763,867 | ______ |
| CC-Dice Loss | ______ | ______ | ______ | ______ | 5,763,867 | ______ |
| PM-Dice Loss (γ=2) | ______ | ______ | ______ | ______ | 5,763,867 | ______ |
| BCE+CC-Dice | ______ | ______ | ______ | ______ | 5,763,867 | ______ |
| BCE+PM-Dice | ______ | ______ | ______ | ______ | 5,763,867 | ______ |
| Full Combined | ______ | ______ | ______ | ______ | 5,763,867 | ______ |

---

## 5. Discussion

### 5.1 Analysis per Dimension

**V1: Loss Function.**
[TO BE FILLED — Did boundary loss improve HD95? Was there a trade-off between λb and volume Dice? Which λb was optimal?]

**V2: Architecture.**
[TO BE FILLED — Did explicit edge branches outperform the baseline in HD95 and Lesion Recall? Was concat consistently better than add? Did Laplacian outperform Sobel? Did FGFE's learnable frequency attention outperform fixed edge operators? Did HF Boundary's dedicated decoder path help?]

**SLA-FB: Data & Loss.**
[TO BE FILLED — Did foreground-aware sampling improve small-lesion metrics? Did CC-Dice or PM-Dice better address the lesion imbalance? Was global Dice still necessary when using lesion-level losses? Did the full combined loss outperform all individual components?]

### 5.2 Cross-Dimension Comparison

[TO BE FILLED — Which single modification gave the largest improvement? Did combining multiple modifications (e.g., V1+V2, V2+SLA-FB) yield additive gains?]

### 5.3 Qualitative Analysis

[TO BE FILLED — Reference figures in `comprehensive_results/qualitative/`. Show cases where baseline misses small satellite lesions but enhanced models detect them. Show boundary refinement in zoomed regions.]

### 5.4 Limitations

- Single-center dataset (BraTS2020); external validation needed
- 3D patch-based training may miss global context for very large tumors
- GradCAM analysis pending (requires GPU with medcam)
- LGG-specific performance not separately analyzed (35.5% LGG cases have zero ET)

---

## 6. Conclusion

[TO BE FILLED — Summarize the best method, key findings, and practical recommendations for clinical tumor segmentation.]

---

## Appendix A: Complete Per-Model Descriptions

| # | Model Label | Category | Architecture | Loss | Key Modification |
|---|---|---|---|---|---|
| 1 | Baseline (BCEDice) | Baseline | ResUNet3d | BCEDiceLoss | — |
| 2 | λb=0.1 | V1: Loss | ResUNet3d | Dice+CE+0.1·BD | Boundary loss (Kervadec 2019) |
| 3 | λb=0.3 | V1: Loss | ResUNet3d | Dice+CE+0.3·BD | Stronger boundary weight |
| 4 | λb=0.5 | V1: Loss | ResUNet3d | Dice+CE+0.5·BD | Max boundary weight |
| 5 | Edge (Sobel, concat) | V2: Architecture | ResUNetEdge | BCEDiceLoss | Sobel edge branch, concat fusion |
| 6 | Edge (Sobel, add) | V2: Architecture | ResUNetEdge | BCEDiceLoss | Sobel edge branch, add fusion |
| 7 | Edge (Laplacian, concat) | V2: Architecture | ResUNetEdge | BCEDiceLoss | Laplacian edge, concat fusion |
| 8 | FGFE | V2: Architecture | ResUNetFGFE | BCEDiceLoss | Decoder Freq. Enhancement (Yao 2025) |
| 9 | HF Boundary | V2: Architecture | ResUNetHFBoundary | BCEDiceLoss + w·BD | Boundary decoder (w=0.2) |
| 10 | HF Boundary+ | V2: Architecture | ResUNetHFBoundary | BCEDiceLoss + w·BD | Stronger boundary (w=0.3) |
| 11 | FG Sampling | SLA-FB: Data | ResUNet3d | BCEDiceLoss | 4-strategy patch sampling |
| 12 | CC-Dice Loss | SLA-FB: Loss | ResUNet3d | Global Dice + CC-Dice | Component-level Dice |
| 13 | PM-Dice Loss | SLA-FB: Loss | ResUNet3d | Global Dice + PM-Dice | Power-mean modulated Dice |
| 14 | BCE+CC-Dice | SLA-FB: Loss | ResUNet3d | BCE + CC-Dice | Ablation: no global Dice |
| 15 | BCE+PM-Dice | SLA-FB: Loss | ResUNet3d | BCE + PM-Dice | Ablation: no global Dice |
| 16 | Full Combined | SLA-FB: Loss | ResUNet3d | Global + CC + PM | Multi-scale supervision |

## Appendix B: How to Fill In Results

```bash
# Step 1: Run comprehensive evaluation on server
cd /root/autodl-tmp/mri_deep
python scripts/eval_comprehensive.py

# Step 2: Download results
# comprehensive_results/comprehensive_results.csv
# comprehensive_results/paper_table_comprehensive.md
# comprehensive_results/confusion_matrices/
# comprehensive_results/training_curves/
# comprehensive_results/per_class_metrics/
# comprehensive_results/qualitative/

# Step 3: Fill the tables in Section 4 above
# All values are in comprehensive_results.csv — copy-paste into this document

# Step 4: Write analysis sections (5.1-5.3) based on results
```

## References

[1] Menze, B.H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." IEEE TMI, 2015.

[2] Bakas, S., et al. "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge." arXiv:1811.02629, 2018.

[3] Kervadec, H., et al. "Boundary loss for highly unbalanced segmentation." MIDL, 2019.

[4] Yao, L., et al. "BraTS-UMamba: Unleashing the Power of Mamba for Brain Tumor Segmentation." MICCAI, 2025.

[5] Zhao, L., et al. "STSNet: a novel 2.5D shallow transformer Siamese network for small object detection." Scientific Reports, 2025.

[6] Hosseini, S.M.H., et al. "PM-Dice: Power Mean Dice Loss for Medical Image Segmentation." arXiv, 2025.

---

*Last updated: 2026-08-11 | 16 models | 20 metrics | Awaiting comprehensive evaluation results*
