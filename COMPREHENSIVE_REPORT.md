# BraTS2020 ET Lesion Analysis & Baseline Model Evaluation — Comprehensive Report

> 生成日期: 2026-08-02  
> 项目: ResUNet Enhancement for Brain Tumor Segmentation  
> 数据: BraTS2020 Training Set (368 cases, 981 ET lesions)  
> 基线: UNet3d, ResUNet3d, AttUNet3d, nnUNet3d

---

## 目录

1. [数据集描述与 ET 病灶统计](#1)
2. [多灶性 ET 可视化验证](#2)
3. [4 基线模型标准指标对比](#3)
4. [4 基线模型高级指标对比](#4)
5. [边界质量分析 (HD95)](#5)
6. [病灶级检测分析 (Lesion-wise Recall)](#6)
7. [代码清单](#7)
8. [下载清单](#8)

---

<h2 id="1">1. 数据集描述与 ET 病灶统计</h2>

### 1.1 数据概览

| 指标 | 值 |
|---|---|
| 总病例数 | 368 |
| HGG | 292 (79.3%) |
| LGG | 76 (20.7%) |
| 有 ET 的病例 | 341 (92.7%) |
| 零 ET 病例 | 27 (7.3%, 全部 LGG) |
| ET 独立病灶总数 | 981 |

### 1.2 病灶大小分布

```
        病灶大小分布 (n=981)
        ─────────────────
  113 vox  ← 中位数! 半数病灶小于这个值

    <50 vox:  412 (42.0%) ████████████████████████
  50-500:     183 (18.6%) ███████████
 500-5000:    115 (11.8%) ███████
5000-50000:   245 (25.0%) ███████████████
   >50000:     26 (2.7%)  ██
```

**关键发现:** 42% 的 ET 病灶小于 50 voxels (~0.05 cm³)。传统 Dice 评分被少数大病灶主导，这些微型病灶在 Dice 中完全不可见。

### 1.3 多灶性统计

| 病灶数 | 病例数 | 占比 |
|---|---|---|
| 0 (无 ET) | 27 | 7.3% |
| 1 (单病灶) | 156 | 42.4% |
| 2 | 83 | 22.6% |
| 3 | 40 | 10.9% |
| 4 | 22 | 6.0% |
| 5-9 | 26 | 7.1% |
| ≥10 | 14 | 3.8% |

**50.3% 的病例有 ≥2 个独立 ET 病灶。多灶性是常态，而非例外。**

### 1.4 HGG vs LGG

| 指标 | HGG (n=292) | LGG (n=76) |
|---|---|---|
| ET voxels (mean) | 23,263 | 5,463 |
| ET/WT ratio | 0.241 | **0.053** (4.6× 差距) |
| 零 ET 比例 | 0% | **35.5%** |
| 多灶性比例 | 50.7% | 48.7% |

### 1.5 论文可用核心数字

| # | 陈述 |
|---|---|
| 1 | 42.0% of ET lesions are < 50 voxels, invisible to conventional Dice |
| 2 | 50.3% of BraTS2020 cases exhibit multi-focal ET |
| 3 | 75.5% follow "dominant lesion + satellite fragments" pattern (LargestRatio ≥ 0.90) |
| 4 | LGG ET/WT ratio is 4.6× lower than HGG (0.053 vs 0.241) |
| 5 | 35.5% of LGG have zero ET — model must learn when NOT to predict |
| 6 | Median ET lesion size: 113 voxels (~0.11 cm³) |

---

<h2 id="2">2. 多灶性 ET 可视化验证</h2>

### 2.1 验证方法

```
方法: 3D 26-邻接连通域标记 (scipy.ndimage.label)
输入: GT 标注中 label=4 (ET only)
过滤: <10 voxels 排除标注噪声

证明策略:
  1. 3D 交互视图 — 每个病灶不同颜色, 旋转可见空间分离
  2. Z 轴滑块 — 逐切片检查颜色是否融合
     (两个连通的病灶必须在某个中间切片上相遇)
  3. 深度-空间散点图 — Z vs X, 不同颜色簇在不同深度
```

### 2.2 选例

| 编号 | Case ID | 类型 | 病灶数 | 用途 |
|---|---|---|---|---|
| A | BraTS20_Training_225 | 单病灶大 ET | 1 | 简单情况的 baseline |
| B | BraTS20_Training_274 | 经典多灶性 | 9 | **Fig 1: Motivation** |
| C | BraTS20_Training_293 | 极度碎片化 | 35 | 问题的严重性 |
| D | BraTS20_Training_329 | LGG 零 ET | 0 | 模型不应 hallucinate |
| E | BraTS20_Training_284 | 微型多灶性 | 16 | baseline 全漏的场景 |

### 2.3 生成的可视化

每例 2 个 HTML 文件（浏览器打开可交互）:
- `{case}_3d.html` — 可旋转 3D 视图，按钮筛选大/小病灶
- `{case}_slices.html` — Z 轴滑块逐切片检查

---

<h2 id="3">3. 4 基线模型标准指标对比</h2>

### 3.1 原有指标 (Dice, IoU)

| 指标 | UNet | ResUNet | AttUNet | nnUNet | Best |
|---|---|---|---|---|---|
| WT Dice | 0.8593 | **0.8860** | 0.8409 | 0.8609 | ResUNet |
| TC Dice | 0.7720 | **0.8175** | 0.7504 | 0.7727 | ResUNet |
| ET Dice | 0.7494 | **0.7585** | 0.7317 | 0.7584 | ResUNet |

**ResUNet 在所有 Dice 指标上领先，选为其作为改进基线是合理的。**

---

<h2 id="4">4. 4 基线模型高级指标对比</h2>

### 4.1 Per-class Recall & Precision

| 指标 | UNet | ResUNet | AttUNet | nnUNet | 含义 |
|---|---|---|---|---|---|
| ET Recall | 0.7700 | 0.7775 | 0.7392 | **0.7974** | 真实 ET 被检出的比例 |
| ET Precision | 0.7888 | 0.7825 | **0.7916** | 0.7550 | 预测 ET 中正确的比例 |
| TC Recall | — | — | — | — | — |
| TC Precision | — | — | — | — | — |

**发现:** nnUNet 检出能力最强 (Recall 0.80)，但假阳性偏高 (Precision 0.76)。ResUNet 两者平衡。

### 4.2 病灶级检测 (Lesion-wise Metrics)

| 指标 | UNet | ResUNet | AttUNet | nnUNet | 含义 |
|---|---|---|---|---|---|
| Lesion Recall | 0.7393 | 0.7181 | 0.7054 | **0.7944** | 独立病灶检出率 |
| Lesion Precision | 0.6535 | 0.8078 | 0.8343 | **0.9123** | 预测病灶中真实比例 |
| Overall Lesion Recall | 0.5847 | 0.5508 | 0.5254 | **0.6525** | 全局病灶检出率 |
| Small-case ET Dice | **0.6357** | 0.6212 | 0.5774 | 0.6261 | 小 ET 病例 Dice |

**核心发现:** 即使最好的 nnUNet，全局病灶召回率仅 65.3%——**1/3 的 ET 病灶被漏检。** 这是论文的核心 motivation。

### 4.3 Small-case vs Overall Dice 落差

| 模型 | Overall ET Dice | Small-case ET Dice | 落差 |
|---|---|---|---|
| UNet | 0.749 | 0.636 | **-11.3%** |
| ResUNet | 0.759 | 0.621 | **-13.8%** |
| AttUNet | 0.732 | 0.577 | **-15.5%** |
| nnUNet | 0.758 | 0.626 | **-13.2%** |

**大病灶主导了 Overall Dice，掩盖了小病灶上的系统性失效。**

---

<h2 id="5">5. 边界质量分析 (HD95)</h2>

| 指标 | UNet | ResUNet | AttUNet | nnUNet | 含义 |
|---|---|---|---|---|---|
| ET HD95 (mm) | **9.50** | 10.26 | 10.21 | 10.91 | 越低越好 |
| TC HD95 (mm) | 10.48 | **9.02** | 11.85 | 12.05 | 越低越好 |

**Trade-off:** UNet 边界最精准 (ET HD95 最低)，nnUNet 检出最强但边界最粗。你的改进应同时优化两者。

---

<h2 id="6">6. 病灶级分析总结</h2>

### 6.1 论文核心叙事

```
           Observation                    Method                  Evidence
           ──────────                    ──────                  ────────
  42% lesions <50 vox,            →  Lesion-wise Recall      → Table comparing
  invisible to Dice                    as primary metric         all 4 baselines

  50% cases multi-focal,          →  Multi-focal ET           → 5-case visualization
  Dice dominated by main lesion       visualization proof        (interactive HTML)

  Small-case Dice 12-15% lower    →  Small-case Dice as      → Ablation table
  than Overall Dice                   secondary metric           (λb experiments)

  UNet best HD95,                 →  Boundary Loss +         → λb=0.1/0.3/0.5
  nnUNet best Lesion Recall           frequency module           experiments
```

### 6.2 消融实验设计

| 实验 | 方法 | 主要对比指标 |
|---|---|---|
| V1 | Dice + CE + Boundary Loss (λb=0.1, 0.3, 0.5) | ET/TC Dice, ET/TC HD95 |
| V2 | 高频边缘辅助分支 (Sobel → decoder) | ET Recall, Lesion Recall, boundary viz |
| V3 | 频域门控融合模块 (LF + HF gate) | All metrics, ablation table |

---

<h2 id="7">7. 代码清单</h2>

### 核心模块
| 文件 | 功能 |
|---|---|
| `models/resunet3d.py` | ResUNet3d 基线架构 |
| `losses/enhanced.py` | Kervadec 2019 BD Loss + DiceCEBoundaryLoss |
| `training/trainer.py` | Trainer (含早停 Patience=25) |
| `evaluation/advanced_metrics.py` | HD95, Lesion-wise Recall, Small-case Dice, 边界可视化 |
| `evaluation/advanced_metrics.py` > `compute_all_advanced_metrics()` | 一键评估入口 |

### 实验脚本
| 文件 | 功能 |
|---|---|
| `scripts/train_enhanced.py` | λb 调优训练 |
| `scripts/et_statistics.py` | 全量 ET 病灶统计 |
| `scripts/visualize_lesions.py` | 交互式多灶性可视化 (Plotly HTML) |

### 评估 Notebook
| 文件 | 功能 |
|---|---|
| `notebooks/experiment_lambda_results.ipynb` | λb 实验对比评估 |
| `notebooks/evaluate_baselines.ipynb` | 4 基线模型一键评估 |

---

<h2 id="8">8. 从服务器下载清单</h2>

```bash
# 在服务器上打包
cd /root/autodl-tmp && tar -czf brats2020_report.tar.gz \
  et_statistics.csv \
  et_components_detail.csv \
  mri_deep/lesion_verification_figures/ \
  mri_deep/evaluation/ \
  mri_deep/models/ \
  mri_deep/losses/ \
  mri_deep/training/ \
  mri_deep/scripts/ \
  mri_deep/notebooks/ \
  mri_deep/RESEARCH_RECORDS.md \
  mri_deep/CHANGELOG.md \
  mri_deep/REPORT_ET_LESION_ANALYSIS.md
```

通过 AutoDL JupyterLab 文件管理器下载 `brats2020_report.tar.gz`。

### 文件说明

| 文件/目录 | 内容 |
|---|---|
| `et_statistics.csv` | 368 病例 × 每例 ET 体素/病灶数/多灶性等 13 列 |
| `et_components_detail.csv` | 981 病灶 × 每病灶所属病例/体积/占比等 6 列 |
| `lesion_verification_figures/` | 5 病例 × 2 HTML (3D + Z-slider) 共 10 个文件 |
| `evaluation/` | 高级评估指标模块 |
| `models/` `losses/` `training/` | 重构后的模块化代码 |
| `scripts/` | 训练/统计/可视化脚本 |
| `notebooks/` | 评估 Jupyter Notebook |
| `RESEARCH_RECORDS.md` | 完整研究数据记录 |
| `REPORT_ET_LESION_ANALYSIS.md` | ET 病灶分析内部报告 |

---

*最后更新: 2026-08-02*
*下一阶段: V1 λb 调优实验评估, V2 高频边缘分支*
