# BraTS2020：4基线模型高级评估 + ET病灶统计 — 综合报告

> 生成日期: 2026-08-02  
> 项目: ResUNet Enhancement for Brain Tumor Segmentation  
> 数据: BraTS2020 Training Set (368 cases, 981 ET lesions)  
> 基线模型: UNet3d, ResUNet3d, AttUNet3d, nnUNet3d  
> 评估工具: `evaluation/advanced_metrics.py` → `compute_all_advanced_metrics()`

---

## 目录

1. [4基线模型完整评估结果](#1)
2. [各指标定义与解读](#2)
3. [核心发现与模型诊断](#3)
4. [数据集ET病灶统计](#4)
5. [多灶性可视化验证](#5)
6. [论文Framing](#6)
7. [代码清单](#7)
8. [下载清单](#8)

---

<h2 id="1">1. 4基线模型完整评估结果</h2>

### 1.1 完整指标对比表

> **数据来源:** BraTS2020 test set（独立划分，random_state=10），4模型均使用各自最低val_loss checkpoint评估，阈值0.33。

```
====================================================================================================
ADVANCED METRICS COMPARISON
====================================================================================================
Metric                         UNet       ResUNet      AttUNet       nnUNet      Best
----------------------------------------------------------------------------------------------------
ET_Dice_mean                  0.7494      0.7585       0.7317       0.7584      ResUNet / nnUNet
ET_Recall_mean                0.7700      0.7775       0.7392       0.7974      nnUNet
ET_Precision_mean             0.7888      0.7825       0.7916       0.7550      AttUNet
ET_HD95_mean (mm)             9.5045     10.2594      10.2126      10.9144      UNet  (lower)
TC_Dice_mean                  0.7720      0.8175       0.7504       0.7727      ResUNet
TC_HD95_mean (mm)            10.4829      9.0201      11.8479      12.0490      ResUNet (lower)
WT_Dice_mean                  0.8593      0.8860       0.8409       0.8609      ResUNet
Lesion_Recall_mean            0.7393      0.7181       0.7054       0.7944      nnUNet
Lesion_Precision_mean         0.6535      0.8078       0.8343       0.9123      nnUNet
Overall_lesion_recall         0.5847      0.5508       0.5254       0.6525      nnUNet
Small_case_ET_Dice_mean       0.6357      0.6212       0.5774       0.6261      UNet
====================================================================================================
备注: HD95 越低越好 (lower is better)，其余指标越高越好 (higher is better)
Small_case = bottom 25% ET volume subset
```

### 1.2 Small-case vs Overall ET Dice 落差

| 模型 | Overall ET Dice | Small-case ET Dice | 落差 |
|---|---|---|---|
| UNet | 0.749 | 0.636 | **-11.3%** |
| ResUNet | 0.759 | 0.621 | **-13.8%** |
| AttUNet | 0.732 | 0.577 | **-15.5%** |
| nnUNet | 0.758 | 0.626 | **-13.2%** |

**大病灶主导了 Overall Dice，掩盖了小病灶上的系统性失效。**

---

<h2 id="2">2. 各指标定义与解读</h2>

### 2.1 标准分割指标（像素级）

| 指标 | 公式 | 解读 |
|---|---|---|
| **ET / TC / WT Dice** | Dice = 2 × |P ∩ T| / (|P| + |T|) | 预测区域与真实区域的重叠度。BraTS标准主指标。**局限性：大病灶主导数值，小病灶贡献极小。** |
| **ET / TC Recall** | Recall = TP / (TP + FN) | 真实肿瘤体素中被模型正确检出的比例。**低Recall = 漏检严重。** 衡量模型"找全了没有"。 |
| **ET / TC Precision** | Precision = TP / (TP + FP) | 模型预测为肿瘤的体素中真正是肿瘤的比例。**低Precision = 模型过度预测（假阳性多）。** 衡量模型"找对了没有"。 |

### 2.2 边界质量指标

| 指标 | 定义 | 解读 |
|---|---|---|
| **ET / TC HD95** | 95th percentile Hausdorff Distance (mm) | 取预测mask表面点到GT mask表面点的所有距离中第95百分位值，对称计算 max(HD(Pred→GT), HD(GT→Pred))。单位mm。**越低说明预测边界越贴近真实边界。** 实现：`scipy.ndimage.distance_transform_edt`（向量化距离变换，无需额外依赖）。 |

### 2.3 病灶级指标（Lesion-wise Metrics）⭐ 论文核心贡献指标

| 指标 | 定义 | 计算方式 | 解读 |
|---|---|---|---|
| **Lesion_Recall_mean** | 每个病例的独立病灶检出率，取所有病例的平均 | per-case: (被检出的GT病灶数)/(GT总病灶数) → mean over cases | 模型是否**找到了**每一个独立的ET病灶？连通域级别的检出率。一个50000vox和1个50vox的小病灶在此指标中**权重相同**。 |
| **Lesion_Precision_mean** | 每个病例的预测病灶准确率，取平均 | per-case: (与GT重叠的Pred病灶数)/(Pred总病灶数) → mean over cases | 模型是否**过度分割**出假病灶？高Precision = 预测的病灶基本都是真的。 |
| **Overall_lesion_recall** | 全数据集的全局病灶检出率 | (所有GT病灶被检出的总数)/(所有病例的GT病灶总数) | 全局视角，所有病灶权重平等（不同于per-case mean，每个病灶而非每个病例权重相同）。|
| **Small_case_ET_Dice** | 仅在小ET病例上的ET Dice | 取数据集ET体积bottom 25%的病例子集，在该子集上计算均值ET Dice | 衡量模型在**最难的小病灶病例**上的表现。与大病灶Dice的落差直接暴露Dice的局限性。 |

#### Lesion-wise Recall 计算细节

```
步骤 1: scipy.ndimage.label(GT_ET, 26-connectivity) → N_GT 个 GT 病灶
步骤 2: scipy.ndimage.label(Pred_ET, 26-connectivity) → N_Pred 个 Pred 病灶  
步骤 3: 过滤掉 < 10 voxels 的碎片（排除标注噪声）
步骤 4: 对每个 GT 病灶，检查是否有任意 Pred 病灶与其有空间重叠（Dice > 0）
步骤 5: Lesion Recall = 被检出的 GT 病灶数 / N_GT
         Lesion Precision = (与GT重叠的Pred病灶数) / N_Pred
```

#### 为什么需要 Lesion-wise Recall（而非只看 Dice）

```
病例 274: GT 有 9 个 ET 病灶 (主病灶 88,587 vox + 8 个小碎片 13-420 vox)
         
模型预测: 找到主病灶 (Dice = 0.92)
          漏掉 6 个小碎片
          
Overall ET Dice ≈ 0.85  ← "看起来很好"
Lesion Recall = 3/9 = 0.33  ← "但漏掉了 2/3 的病灶！"
```

**这就是Dice的局限性：** 大病灶贡献了绝大多数体素，小病灶在Dice中完全不可见。Lesion-wise Recall 让每个病灶权重平等。

---

<h2 id="3">3. 核心发现与模型诊断</h2>

### 3.1 各模型定位

| 模型 | 定位 | 强项 | 弱项 |
|---|---|---|---|
| **ResUNet** | 综合最优 | TC Dice 0.818, TC HD95 9.02, WT Dice 0.886 | Lesion Recall 0.72 (会漏病灶) |
| **nnUNet** | 病灶检出之王 | Lesion Recall 0.79, Lesion Precision 0.91, ET Recall 0.80 | ET HD95 10.91 (边界最粗), ET Precision 0.76 (假阳性多) |
| **UNet** | 边界最精准 | ET HD95 9.50 (最贴近GT), Small-case Dice 0.64 | Lesion Recall 0.74 (中等) |
| **AttUNet** | 全面落后 | — | 几乎所有指标垫底，CBAM+AG在3D小病灶场景效果有限 |

### 3.2 关键Trade-off

```
nnUNet:   Lesion Recall 0.79  ←→  ET HD95 10.91  (高检出，粗边界)
UNet:     ET HD95 9.50       ←→  Lesion Recall 0.74  (精准边界，漏病灶)
ResUNet:  TC Dice 0.818      ←→  Lesion Recall 0.72  (中间位置)

→ 改进目标: 同时提升 Lesion Recall 和 HD95，打破这个 trade-off
```

### 3.3 为什么选 ResUNet 作为改进基线

- **3项Dice + TC HD95 最优** — 综合分割能力最强
- **Lesion Recall 仅 0.72** — 病灶检出有明确改进空间
- **Small-case Dice 比 Overall 低 13.8%** — 小病灶是明确短板
- **架构有残差连接** — 天然适合加模块（残差路径可直接注入新特征）

---

<h2 id="4">4. 数据集 ET 病灶统计</h2>

### 4.1 核心数字

| 指标 | 值 |
|---|---|
| 总病例数 | 368 |
| HGG / LGG | 292 (79.3%) / 76 (20.7%) |
| ET独立病灶总数 | 981 |
| 多灶性比例 (>=2病灶) | 50.3% (185/368) |
| 病灶体积中位数 | 113 voxels (~0.11 cm³) |
| 微型病灶 (<50 vox) 占比 | 42.0% (412/981) |
| 零ET病例 (全部LGG) | 27 (7.3%) |

### 4.2 论文可用核心数字

| 陈述 | 数值 |
|---|---|
| 微型病灶 (<50 vox) 占全部病灶比例 | 42.0% |
| 多灶性ET病例比例 | 50.3% |
| "大病灶+卫星碎片"模式占比（LargestRatio >= 0.9）| 75.5% |
| LGG 零ET比例 | 35.5% |
| ET病灶体积中位数 | 113 voxels |
| 所有模型Lesion Recall上限 | 65.3% (nnUNet) |

---

<h2 id="5">5. 多灶性可视化验证</h2>

### 5.1 验证方法

```
方法: 3D 26-邻接连通域标记 (scipy.ndimage.label)
输入: GT标注中 label=4 (ET only, BraTS标准标注协议)
过滤: < 10 voxels 碎片排除标注噪声

证明策略 (交互式HTML):
  1. 3D 视图 (_3d.html): 每个病灶不同颜色，旋转可见空间分离
  2. Z轴滑块 (_slices.html): 逐切片检查颜色是否融合
     (两个连通的病灶必然在某个中间切片上相遇 → 颜色会融合)
     如果滑块切过所有切片，两个颜色从未融合 → 确凿不连通
```

### 5.2 选例

| 编号 | Case ID | 类型 | 病灶数 | 论文用途 |
|---|---|---|---|---|
| A | BraTS20_Training_225 | 单病灶大ET | 1 (111K vox) | 简单情况的baseline |
| B | BraTS20_Training_274 | 经典多灶性 | 9 (主+8碎片) | **Fig 1: Motivation** |
| C | BraTS20_Training_293 | 极度碎片化 | 35 | 问题的严重性 |
| D | BraTS20_Training_329 | LGG零ET | 0 | 不应hallucinate ET |
| E | BraTS20_Training_284 | 微型多灶性 | 16 (全<2800vox) | baseline全漏的场景 |

### 5.3 输出文件

每例2个HTML（浏览器打开可交互）:
- `{case}_3d.html` — 可旋转3D视图，按钮切换显示大小病灶
- `{case}_slices.html` — Z轴滑块逐切片检查连通性

生成脚本: `scripts/visualize_lesions.py`

---

<h2 id="6">6. 论文 Framing</h2>

```
Observation:          42% of ET lesions < 50 vox, invisible to Dice
Evidence A:           Small-case Dice 12-15% lower than Overall Dice (all 4 models)
Evidence B:           All models' Overall Lesion Recall <= 65%
Evidence C:           50% cases multi-focal, 75% follow "main+satellites" pattern
Gap:                  Existing methods optimize pixel-level Dice,
                      systematically missing small satellite lesions

Proposed Method:      Targets BOTH pixel accuracy (Dice, HD95)
                      AND lesion-level detection (Lesion Recall, Small-case Dice)
```

### 消融实验设计

| 实验 | 方法 | 主要对比指标 |
|---|---|---|
| V1 | Dice + CE + Boundary Loss (λb=0.1, 0.3, 0.5) | ET/TC Dice, ET/TC HD95, boundary viz |
| V2 | 高频边缘辅助分支 (Sobel → decoder) | ET Recall, Lesion Recall, HD95 |
| V3 | 频域门控融合模块 (LF + HF gate) | All metrics, ablation table |

---

<h2 id="7">7. 代码清单</h2>

### 核心模块
| 文件 | 功能 |
|---|---|
| `models/resunet3d.py` | ResUNet3d 基线架构 |
| `losses/enhanced.py` | Kervadec 2019 BD Loss + DiceCEBoundaryLoss |
| `training/trainer.py` | Trainer（含早停 patience=25, min_delta=1e-4）|
| `evaluation/advanced_metrics.py` | HD95, Lesion-wise Recall, Small-case Dice, print_comparison_table, boundary_overlay |

### 实验脚本
| 文件 | 功能 |
|---|---|
| `scripts/train_enhanced.py` | λb 调优训练 |
| `scripts/et_statistics.py` | 全量ET病灶统计（输出两个CSV）|
| `scripts/visualize_lesions.py` | 交互式多灶性可视化（Plotly HTML）|

### 评估 Notebook
| 文件 | 功能 |
|---|---|
| `notebooks/evaluate_baselines.ipynb` | 4基线模型一键评估 |
| `notebooks/experiment_lambda_results.ipynb` | λb实验对比评估 |

### 关键函数入口
```python
from evaluation.advanced_metrics import compute_all_advanced_metrics, print_comparison_table

# 评估单个模型
metrics = compute_all_advanced_metrics(model, test_dataloader, model_name='ResUNet')

# 对比多个模型
print_comparison_table([unet_m, resunet_m, attunet_m, nnunet_m])
```

---

<h2 id="8">8. 下载清单</h2>

```bash
# 服务器上打包
cd /root/autodl-tmp && tar -czf brats2020_report.tar.gz \
  et_statistics.csv \
  et_components_detail.csv \
  mri_deep/lesion_verification_figures/ \
  mri_deep/COMPREHENSIVE_REPORT.md \
  mri_deep/RESEARCH_RECORDS.md \
  mri_deep/CHANGELOG.md \
  mri_deep/evaluation/ \
  mri_deep/models/ \
  mri_deep/losses/ \
  mri_deep/training/ \
  mri_deep/data/ \
  mri_deep/scripts/ \
  mri_deep/notebooks/
```

| 文件 | 内容 |
|---|---|
| `COMPREHENSIVE_REPORT.md` | **本报告** — 论文写作直接参考 |
| `et_statistics.csv` | 368病例×13列 ET病灶统计 |
| `et_components_detail.csv` | 981病灶×6列 病灶明细 |
| `lesion_verification_figures/*.html` | 10个交互式可视化 |
| `evaluation/` `models/` `losses/` `training/` | 重构后模块化代码 |
| `scripts/` | 训练/统计/可视化脚本 |
| `CHANGELOG.md` | 完整修改历史 |

---

*最后更新: 2026-08-02*
