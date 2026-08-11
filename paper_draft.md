# ResUNet 增强实验报告 — BraTS2020 脑肿瘤分割

> 生成日期: 2026-08-11
> 项目: ResUNet Enhancement for Brain Tumor Segmentation
> 数据: BraTS2020 Training Set (368 cases, 981 ET lesions)
> 基线模型: ResUNet3d (n_channels=24, 5,763,867 params)
> 实验模型: 15 个（V1 Loss × 3, V2 Architecture × 6, SLA-FB Data × 1, SLA-FB Loss × 5）

---

## 目录

1. [实验背景与动机](#1)
2. [实验设计总览](#2)
3. [各模型改动详解](#3)
4. [评估指标体系](#4)
5. [实验结果](#5)
6. [分析与讨论](#6)
7. [下一步计划](#7)

---

<h2 id="1">1. 实验背景与动机</h2>

### 1.1 问题陈述

BraTS2020 数据集中 Enhancing Tumor (ET) 的病灶分布极不均衡：

| 发现 | 数值 |
|---|---|
| 微型病灶 (<50 voxels) 占比 | **42.0%** (412/981) |
| 多灶性病例 (≥2 disconnected lesions) | **50.3%** (185/368) |
| "大病灶+卫星碎片"模式 | **75.5%** 的病例 |
| ET 病灶体积中位数 | **113 voxels** (~0.11 cm³) |
| ET 仅占全脑体素的 | **~0.5%** |

**核心矛盾：** Overall Dice 由大病灶主导，掩盖了小病灶上的系统性失效。Baseline ResUNet 的 Overall ET Dice ≈ 0.76，但 Lesion-wise Recall 仅 0.72，Small-case ET Dice 比 Overall 低 13.8%。

### 1.2 实验目标

从三个维度系统性地改进 ResUNet baseline，打破"大病灶 Dice 好看、小病灶全漏"的 trade-off：

| 维度 | 变量 | 方法 |
|---|---|---|
| **V1: Loss 函数** | 损失函数设计 | Dice + CE + Boundary Loss (Kervadec 2019) |
| **V2: 模型架构** | 网络结构 | Sobel/Laplacian 边缘分支, FGFE 频域增强, HF Boundary |
| **SLA-FB: 数据+Loss** | 数据采样 + 病灶级损失 | FG-aware patch sampling, CC-Dice, PM-Dice |

**公平性原则：每个实验只改一个变量。** 数据划分、超参数、早停标准全部统一。

---

<h2 id="2">2. 实验设计总览</h2>

### 2.1 统一训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| 架构 | ResUNet3d (n_channels=24) | 所有实验统一 |
| 优化器 | Adam, lr=5e-4 | — |
| 学习率调度 | ReduceLROnPlateau, patience=2 | val_loss 不降则减半 |
| Batch size | 1, gradient accumulation=4 | 3D patch 显存限制 |
| 早停 | patience=25, min_delta=1e-4 | 监控 val_loss |
| 数据划分 | train_test_split(random_state=10, test_size=0.3) | 所有模型同一划分 |
| 预处理 | crop [40:210, 40:210, 20:120] + min-max normalize | — |
| 推理阈值 | 0.33 | 统一二值化阈值 |

### 2.2 16 模型一览

```
ResUNet Baseline (BCEDiceLoss)
│
├── V1: 改 Loss 函数 ─────────────────────────────
│   ├── λb=0.1 (Dice+CE+0.1·BD)
│   ├── λb=0.3 (Dice+CE+0.3·BD)
│   └── λb=0.5 (Dice+CE+0.5·BD)
│
├── V2: 改模型架构 ─────────────────────────────
│   ├── Edge (Sobel, concat)          ← 1st导数边缘, concat融合
│   ├── Edge (Sobel, add)             ← 1st导数边缘, add融合
│   ├── Edge (Laplacian, concat)      ← 2nd导数边缘, concat融合
│   ├── FGFE (Freq. Enhancement)      ← decoder特征层频域增强
│   ├── HF Boundary (Laplacian, w=0.2) ← 边界解码器分支
│   └── HF Boundary+ (Laplacian, w=0.3)← 更强边界监督
│
└── SLA-FB: 改数据采样 + 病灶级 Loss ──────────
    ├── FG Sampling (4-strategy)       ← 前景感知patch采样
    ├── CC-Dice Loss                   ← 连通域级Dice
    ├── PM-Dice Loss (γ=2)            ← Power-Mean Dice
    ├── BCE+CC-Dice (no Global)       ← 消融: 无全局Dice
    ├── BCE+PM-Dice (no Global)       ← 消融: 无全局Dice
    └── Full Combined (Global+CC+PM)  ← 多尺度联合监督
```

---

<h2 id="3">3. 各模型改动详解</h2>

### 3.1 Baseline: ResUNet3d + BCEDiceLoss

**不改动。** 标准 3D ResUNet，作为所有实验的对照基线。

| 组件 | 规格 |
|---|---|
| 编码器 | ResBlock (Conv3d→GroupNorm→ReLU)×2 + 残差连接 |
| 下采样 | MaxPool3d (kernel=2) |
| 解码器 | ResBlock + Trilinear上采样 + Skip Connection |
| 输出 | Conv3d(24→3) + Sigmoid |
| Loss | BCEDiceLoss = BCE + Dice |
| 参数量 | 5,763,867 |

---

### 3.2 V1: Loss 函数消融 (3 个模型)

**改动：** 只换 Loss 函数，模型架构不变。Warm-start 从 baseline epoch 199 checkpoint。

**Loss 公式：**

$$L = 1.0 \cdot L_{Dice} + 0.5 \cdot L_{CE} + \lambda_b \cdot L_{Boundary}$$

| 模型 | λb | Class Weights | 改动说明 |
|---|---|---|---|
| **λb=0.1** | 0.1 | WT=1, TC=3, ET=5 | 轻微边界监督，测试 BD loss 的基础效果 |
| **λb=0.3** | 0.3 | WT=1, TC=3, ET=5 | 中等边界权重，预期改善 HD95 |
| **λb=0.5** | 0.5 | WT=1, TC=3, ET=5 | 强边界权重，测试是否过度压制体积分割 |

**Boundary Loss (Kervadec 2019):** 对 GT mask 边界做 distance transform，让模型预测的边界与 GT 边界对齐。只监督边界像素，不与 Dice 冲突。

---

### 3.3 V2: 模型架构改动 (6 个模型)

#### 3.3.1 Edge Branch — Sobel / Laplacian 边缘提取

**核心思路：** 在 ResUNet 基础上增加一个**并行的边缘检测分支**。对原始 4 通道 MRI 做边缘提取，将边缘特征注入 decoder 各层。直觉：MRI 中肿瘤边界的梯度信息是显式的先验知识，模型不应从头学。

| 模型 | 边缘算子 | 融合方式 | 参数量变化 | 改动点 |
|---|---|---|---|---|
| **Edge (Sobel, concat)** | Sobel (1阶导数) | Concat | +300K | 新增边缘分支 |
| **Edge (Sobel, add)** | Sobel (1阶导数) | Add (残差) | +300K | 改融合策略为add |
| **Edge (Laplacian, concat)** | Laplacian (2阶导数) | Concat | +300K | 改边缘算子 |

**Sobel vs Laplacian:**
- Sobel: 1阶导数，检测梯度方向，对噪声较鲁棒
- Laplacian: I−blur(I)，2阶导数，检测零交叉点，捕获更细的边缘

**Concat vs Add:**
- Concat: 保留边缘特征的独立通道，decoder 可以学习选择性使用
- Add: 残差注入，要求边缘特征和 decoder 特征在同一个语义空间

#### 3.3.2 FGFE: 特征层频域增强

**来源:** Yao et al., BraTS-UMamba, MICCAI 2025.

**与 Edge Branch 的关键区别：**

| | Edge Branch | FGFE |
|---|---|---|
| 作用对象 | 原始 MRI 像素 (4通道) | Decoder 特征图 (24-96通道) |
| 机制 | 固定数学算子 (先验知识) | 可学习增强 (带参数 Attention) |
| 维度 | **数据层面** | **特征层面** |

**机制：**
1. Laplacian 金字塔分解: $F \rightarrow F_h$ (高频) + $F_l$ (低频)
2. Cross-Attention: $F_h$ 和 $F_l$ 各自去 query skip-connection 特征 $F_s$
3. 残差融合: $F_{out} = F_s + \text{Conv}([F_h^{attn}, F_l^{attn}])$

**两者不是替代关系，理论上可以组合。**

#### 3.3.3 HF Boundary: 高频边界分支

**思路：** 把 Edge Branch 的思路做到极致——不只注入 edge 特征，而是给边界专门建一个 decoder。

| 组件 | 说明 |
|---|---|
| 边缘提取 | Laplacian (I−blur(I)) |
| 边界解码器 | 独立 decoder path → boundary prediction map |
| 边界监督 | Boundary loss (weight w) |
| 融合方式 | Attention-gated fusion with main decoder |

| 模型 | w | 改动说明 |
|---|---|---|
| **HF Boundary** | 0.2 | 轻量边界监督 |
| **HF Boundary+** | 0.3 | 更强边界监督，测试是否过拟合边界 |

---

### 3.4 SLA-FB: 数据采样 + 病灶级 Loss (6 个模型)

#### 3.4.1 FG Sampling: 前景感知 Patch 采样

**来源:** STSNet (Zhao et al., Scientific Reports 2025)，3D 适配。

**问题：** BraTS2020 中 ET 体素仅占全图的 ~0.5%。随机采样 patch 时 ET 区域几乎不出现 → 模型缺乏 ET 训练信号。

**方法：** 4 策略加权随机采样，训练时在线实时决定 patch 位置：

| 策略 | 采样规则 | 权重 | 对应 STSNet |
|---|---|---|---|
| `random` | 均匀随机 crop | 20% | 标准 DataLoader |
| `foreground` | 以 WT 肿瘤像素为中心 | 30% | TwoStreamBatchSampler primary |
| `et_centered` | 以 ET 连通域质心为中心（每个灶等概率） | 30% | 中心裁剪 |
| `small_lesion` | 只从小 ET 灶 (<50 vox) 质心采样 | 20% | label_count < 1000 |

**核心设计决策：**
- **不做 3D resize 放大**——用采样频率（80% 在 ET 区域）替代物理放大
- **降级链：** small_lesion → et_centered → foreground → random（池空时自动降级）
- **模型/Loss/超参不变**——单变量实验

#### 3.4.2 CC-Dice: 连通域级 Dice Loss

**问题：** 全局 Dice 把所有体素加在一起算一个分数。1 个 100,000-voxel 大病灶 + 8 个 20-voxel 小碎片 → 小碎片在 Dice 中几乎不可见。

**方法：** 对每个 ET 连通域单独算 Dice，然后取平均：

$$L_{CC-Dice} = \frac{1}{K} \sum_{k=1}^{K} \text{Dice}(P_k, T_k)$$

其中 $K$ 为病例中 ET 连通域数量。**每个病灶等权，不管大小。**

#### 3.4.3 PM-Dice: Power-Mean Dice Loss

**来源:** Hosseini et al., 2025.

**方法：** 用 $|y - \hat{p}|^\gamma$ 调制 Dice 中的每一项：

$$L_{PM-Dice} = \text{Dice}(|y-\hat{p}|^\gamma \odot y, |y-\hat{p}|^\gamma \odot \hat{p})$$

γ=2 时，被误分类的像素获得指数级的更高权重。与 CC-Dice 互补：CC-Dice 在**病灶级别**等权，PM-Dice 在**像素级别**加权。

#### 3.4.4 消融实验

| 模型 | Global Dice | CC-Dice | PM-Dice | 消融目的 |
|---|---|---|---|---|
| **CC-Dice** | ✅ | ✅ | — | 测试连通域级 Dice 的基础效果 |
| **PM-Dice (γ=2)** | ✅ | — | ✅ | 测试 Power-Mean 调制的基础效果 |
| **BCE+CC-Dice** | — | ✅ | — | **消融**: 全局 Dice 是否必要？ |
| **BCE+PM-Dice** | — | — | ✅ | **消融**: 全局 Dice 是否必要？ |
| **Full Combined** | ✅ | ✅ | ✅ | 多尺度联合：体积+病灶+像素 |

---

<h2 id="4">4. 评估指标体系</h2>

### 4.1 指标总览（20 项，5 维度）

| 维度 | 指标 | 数量 | 数据来源 |
|---|---|---|---|
| **像素级分类** | Per-class Accuracy, Precision, Recall, F1 × (WT, TC, ET) | 12 | 🆕 本次新增 |
| **区域分割** | Per-class Dice × (WT, TC, ET), Per-class Jaccard × (WT, TC, ET) | 6 | 🆕 本次新增 + 已有 |
| **边界质量** | ET/TC HD95 (mm), ET/TC NSD (τ=1mm) | 4 | ♻️ 已有 |
| **病灶检出** | Lesion-wise Recall, Lesion-wise Precision, Overall Lesion Recall | 3 | ♻️ 已有 |
| **小病灶专项** | Small-case ET Dice (bottom 25% ET volume) | 1 | ♻️ 已有 |
| **效率** | 参数量, 推理时间 | 2 | ♻️ 已有 |

### 4.2 各指标详解

#### 像素级分类指标 (HFF Notebook Cell 70-85)

对每个体素做二分类（肿瘤 vs 背景），分别对 WT、TC、ET 三个类别计算：

| 指标 | 公式 | 解读 |
|---|---|---|
| **Accuracy** | (TP+TN) / (TP+TN+FP+FN) | 总体正确率。**注意：类别极度不平衡(ET ~0.5%)，Accuracy 会虚高。** |
| **Precision** | TP / (TP+FP) | 预测为肿瘤的体素中正确的比例。低 = 假阳性多（模型过度预测）。 |
| **Recall** | TP / (TP+FN) | 真实肿瘤中被检出的比例。低 = 漏检多。 |
| **F1-Score** | 2·P·R / (P+R) | Precision 和 Recall 的调和平均。 |

#### 区域分割指标 (HFF Notebook Cell 98-115)

| 指标 | 公式 | 解读 |
|---|---|---|
| **Dice (F1)** | 2|P∩T| / (|P|+|T|) | BraTS 标准指标。大病灶主导整体数值。 |
| **Jaccard (IoU)** | |P∩T| / (|P∪T|) | 比 Dice 更严格（分母更大）。Dice 的互补指标。 |

**注意：** Dice 和 Jaccard 存在一一对应关系 ($J = D/(2-D)$, $D = 2J/(1+J)$)，但 Jaccard 对小误差更敏感，在评估中保留两者便于对标其他论文。

#### 边界质量指标

| 指标 | 定义 | 解读 |
|---|---|---|
| **HD95** | 95th percentile Hausdorff Distance (mm) | 预测边界到 GT 边界的第 95 百分位距离。**越低越好。** 用 95 分位而非 max 来排除极端异常值。对称计算 max(HD(P→G), HD(G→P))。实现: `scipy.ndimage.distance_transform_edt`。 |
| **NSD (τ=1mm)** | Normalized Surface Dice | 在 τ=1mm 容差范围内的边界匹配率。**越高越好。** 互补于 HD95：HD95 看最差情况，NSD 看整体边界匹配。 |

#### 病灶级指标 ⭐ 核心贡献

| 指标 | 计算方式 | 解读 |
|---|---|---|
| **Lesion-wise Recall (per-case)** | 每个病例: (被检出GT病灶数)/(GT总病灶数) → mean over cases | 模型是否**找到**了每个独立 ET 病灶？50,000-vox 大病灶和 50-vox 小病灶**权重相同**。 |
| **Lesion-wise Precision (per-case)** | 每个病例: (与GT重叠的Pred病灶数)/(Pred总病灶数) → mean over cases | 模型是否**过度分割**出假病灶？ |
| **Overall Lesion Recall (global)** | 所有GT病灶中被检出的总数 / 所有GT病灶总数 | 全局视角，每个病灶（不是每个病例）权重相同。 |

**病灶检出计算细节：**
1. `scipy.ndimage.label(GT_ET, 26-connectivity)` → N_GT 个 GT 病灶
2. `scipy.ndimage.label(Pred_ET, 26-connectivity)` → N_Pred 个 Pred 病灶
3. 过滤 < 10 voxels 碎片（排除标注噪声）
4. 对每个 GT 病灶，检查是否有任意 Pred 病灶与其空间重叠 (Dice > 0)
5. Lesion Recall = 被检出的 GT 病灶数 / N_GT

#### 小病灶专项指标

| 指标 | 计算方式 | 解读 |
|---|---|---|
| **Small-case ET Dice** | 取数据集 ET 体积 bottom 25% 的病例，在该子集上计算 ET Dice 均值 | 暴露 Dice 对小病灶的盲区。与 Overall ET Dice 的落差越大，说明模型越依赖大病灶撑 Dice。 |

---

<h2 id="5">5. 实验结果</h2>

> **填表说明：** 在服务器上运行 `python scripts/eval_comprehensive.py --existing-results all_experiments_results.json`，从 `comprehensive_results/comprehensive_results.csv` 复制数值填入下表。

### 5.1 区域分割指标 — Per-Class Dice & Jaccard

| 模型 | WT Dice | TC Dice | ET Dice | WT Jaccard | TC Jaccard | ET Jaccard |
|---|---|---|---|---|---|---|
| Baseline (BCEDice) | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| λb=0.1 | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| λb=0.3 | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| λb=0.5 | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| Edge (Sobel, concat) | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| Edge (Sobel, add) | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| Edge (Laplacian, concat) | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| FGFE | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| HF Boundary (w=0.2) | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| HF Boundary+ (w=0.3) | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| FG Sampling | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| CC-Dice Loss | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| PM-Dice Loss (γ=2) | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| BCE+CC-Dice | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| BCE+PM-Dice | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |
| Full Combined | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ | ___ ± ___ |

### 5.2 像素级分类指标 — ET Class

| 模型 | Accuracy | Precision | Recall | F1-Score | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| Baseline | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| λb=0.1 | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| λb=0.3 | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| λb=0.5 | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| Edge (Sobel, concat) | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| Edge (Sobel, add) | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| Edge (Laplacian, concat) | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| FGFE | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| HF Boundary | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| HF Boundary+ | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| FG Sampling | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| CC-Dice Loss | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| PM-Dice Loss | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| BCE+CC-Dice | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| BCE+PM-Dice | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| Full Combined | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |

> WT 和 TC 的完整像素分类表见 `comprehensive_results/paper_table_comprehensive.md`

### 5.3 边界质量 & 病灶检出 — 核心指标对比

| 模型 | ET HD95↓ | ET NSD↑ | TC HD95↓ | TC NSD↑ | Lesion Recall↑ | Lesion Prec.↑ | Small ET Dice↑ |
|---|---|---|---|---|---|---|---|
| Baseline | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| λb=0.1 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| λb=0.3 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| λb=0.5 | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| Edge (Sobel, concat) | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| Edge (Sobel, add) | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| Edge (Laplacian, concat) | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| FGFE | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| HF Boundary | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| HF Boundary+ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| FG Sampling | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| CC-Dice Loss | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| PM-Dice Loss | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| BCE+CC-Dice | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| BCE+PM-Dice | ___ | ___ | ___ | ___ | ___ | ___ | ___ |
| Full Combined | ___ | ___ | ___ | ___ | ___ | ___ | ___ |

*↓ = 越低越好; ↑ = 越高越好。每列最优值标 **粗体**。*

### 5.4 训练效率

| 模型 | 总 Epoch | 最优 Epoch | 训练时间/epoch (s) | 验证时间/epoch (s) | 总训练时间 (h) | 参数量 | 推理时间 (s/case) |
|---|---|---|---|---|---|---|---|
| Baseline | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |
| λb=0.1 | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |
| λb=0.3 | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |
| λb=0.5 | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |
| Edge (Sobel, concat) | ___ | ___ | ___ | ___ | ___ | ~6,064K | ___ |
| Edge (Sobel, add) | ___ | ___ | ___ | ___ | ___ | ~6,064K | ___ |
| Edge (Laplacian, concat) | ___ | ___ | ___ | ___ | ___ | ~6,064K | ___ |
| FGFE | ___ | ___ | ___ | ___ | ___ | ~6,264K | ___ |
| HF Boundary | ___ | ___ | ___ | ___ | ___ | ~6,264K | ___ |
| HF Boundary+ | ___ | ___ | ___ | ___ | ___ | ~6,264K | ___ |
| FG Sampling | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |
| CC-Dice Loss | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |
| PM-Dice Loss | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |
| BCE+CC-Dice | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |
| BCE+PM-Dice | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |
| Full Combined | ___ | ___ | ___ | ___ | ___ | 5,763,867 | ___ |

---

<h2 id="6">6. 分析与讨论</h2>

### 6.1 V1: Loss 函数消融分析

**核心问题：** Boundary loss 是否改善了边界质量？是否有 λb 的最优值？边界监督是否会损害体积分割？

**[待填写 — 对比 λb=0.1/0.3/0.5 的 ET HD95、ET Dice、Lesion Recall]**

---

### 6.2 V2: 模型架构分析

**核心问题：** 显式边缘分支是否能同时改善 HD95 和 Lesion Recall？

**子问题：**
- Concat vs Add 融合：哪个更好？为什么？
- Sobel vs Laplacian：1阶导数 vs 2阶导数，哪个对肿瘤边界更有用？
- FGFE 的可学习频域增强是否优于固定的边缘算子？
- HF Boundary 的独立 decoder 路径是否比简单 edge injection 更有效？

**[待填写 — 按子问题逐一分析]**

---

### 6.3 SLA-FB: 数据采样 + 病灶级 Loss 分析

**核心问题：** 在数据层面（FG Sampling）和 Loss 层面（CC-Dice、PM-Dice）分别有什么效果？

**子问题：**
- FG Sampling 是否改善了 Small-case ET Dice 和 Lesion Recall？
- CC-Dice 和 PM-Dice 哪个对小病灶更有效？两者是否互补？
- 消融实验：全局 Dice 是否仍然必要？（BCE+CC-Dice vs CC-Dice）
- Full Combined 是否优于任何单一 Loss？

**[待填写]**

---

### 6.4 跨维度对比

| 维度 | 最优模型 | 最佳 ET Dice | 最佳 ET HD95 | 最佳 Lesion Recall | 最佳 Small ET Dice |
|---|---|---|---|---|---|
| Baseline | ResUNet | ___ | ___ | ___ | ___ |
| V1 (Loss) | ___ | ___ | ___ | ___ | ___ |
| V2 (Architecture) | ___ | ___ | ___ | ___ | ___ |
| SLA-FB (Data) | ___ | ___ | ___ | ___ | ___ |
| SLA-FB (Loss) | ___ | ___ | ___ | ___ | ___ |
| **全局最优** | **___** | **___** | **___** | **___** | **___** |

**核心结论：**

**[待填写 — 哪个单一改动效果最大？不同维度的改进是否可以叠加？是否存在 trade-off（如 ET Dice ↑ 但 HD95 ↓）？]**

---

### 6.5 关键 Trade-off 分析

Baseline 评估中已经暴露了一个 trade-off：nnUNet 的 Lesion Recall 最高但 HD95 最差，UNet 的 HD95 最好但 Lesion Recall 较低。

**[待填写 — 你的各改进模型在这个 trade-off 光谱上的位置？是否有模型同时改善了 Lesion Recall 和 HD95（打破 trade-off）？]**

---

### 6.6 Small-case vs Overall ET Dice 落差

计算每个模型的 Small-case ET Dice 与 Overall ET Dice 的落差：

| 模型 | Overall ET Dice | Small-case ET Dice | 落差 (%) |
|---|---|---|---|
| Baseline | ___ | ___ | ___ |
| 全局最优 | ___ | ___ | ___ |

**[待填写 — 落差是否缩小？哪个方法对缩小落差最有效？]**

---

<h2 id="7">7. 下一步计划</h2>

### 7.1 待完成评估

- [ ] 运行 `python scripts/eval_comprehensive.py --existing-results all_experiments_results.json` 获取完整指标
- [ ] 填写 Section 5 所有表格
- [ ] 写 Section 6 逐维度分析
- [ ] 定性可视化：选 3-5 个典型病例做 slice overlay 对比（已有代码 `--no-figures` 去掉即可）
- [ ] GradCAM XAI 分析（需要 medcam 库，可选）

### 7.2 待探索方向

- [ ] **组合实验：** V1 最优 λb + V2 最优架构（如 Edge concat + Boundary loss）
- [ ] **组合实验：** V2 最优架构 + SLA-FB 最优 Loss
- [ ] **LGG 子集分析：** 35.5% LGG 零 ET，应单独评估
- [ ] **病灶大小分层报告：** <50 / 50-500 / >500 vox 三层 Lesion Recall
- [ ] **外部验证：** BraTS2021 或私有数据

### 7.3 论文写作

- [ ] 确定核心 story：哪个改进维度贡献最大？
- [ ] 选代表性结果图（训练曲线、混淆矩阵、定性 overlay）
- [ ] 写 Method section（可直接翻译本文档 Section 3）
- [ ] 写 Results section（填完 Section 5 后）
- [ ] 写 Discussion（填完 Section 6 后）

---

## 附录 A: 模型速查表

| # | 模型 | 类别 | 架构 | Loss | 核心改动 |
|---|---|---|---|---|---|
| 1 | Baseline | Baseline | ResUNet3d | BCEDiceLoss | — |
| 2 | λb=0.1 | V1 | ResUNet3d | Dice+CE+0.1·BD | Boundary loss (Kervadec 2019) |
| 3 | λb=0.3 | V1 | ResUNet3d | Dice+CE+0.3·BD | 增强边界权重 |
| 4 | λb=0.5 | V1 | ResUNet3d | Dice+CE+0.5·BD | 最大边界权重 |
| 5 | Edge (Sobel, concat) | V2 | ResUNetEdge | BCEDiceLoss | Sobel 边缘分支, concat 融合 |
| 6 | Edge (Sobel, add) | V2 | ResUNetEdge | BCEDiceLoss | Sobel 边缘分支, add 融合 |
| 7 | Edge (Laplacian, concat) | V2 | ResUNetEdge | BCEDiceLoss | Laplacian 边缘, concat 融合 |
| 8 | FGFE | V2 | ResUNetFGFE | BCEDiceLoss | Decoder 频域增强 (Yao 2025) |
| 9 | HF Boundary | V2 | ResUNetHFBoundary | BCEDice + 0.2·BD | 独立边界解码器 |
| 10 | HF Boundary+ | V2 | ResUNetHFBoundary | BCEDice + 0.3·BD | 更强边界监督 |
| 11 | FG Sampling | SLA-FB Data | ResUNet3d | BCEDiceLoss | 4策略前景采样 |
| 12 | CC-Dice Loss | SLA-FB Loss | ResUNet3d | Global Dice + CC-Dice | 连通域级 Dice |
| 13 | PM-Dice Loss | SLA-FB Loss | ResUNet3d | Global Dice + PM-Dice | Power-Mean 调制 Dice |
| 14 | BCE+CC-Dice | SLA-FB Loss | ResUNet3d | BCE + CC-Dice | 消融: 无全局 Dice |
| 15 | BCE+PM-Dice | SLA-FB Loss | ResUNet3d | BCE + PM-Dice | 消融: 无全局 Dice |
| 16 | Full Combined | SLA-FB Loss | ResUNet3d | Global + CC + PM | 多尺度联合监督 |

## 附录 B: 操作步骤

```bash
# 1. 服务器上运行综合评估
cd /root/autodl-tmp/mri_deep && git pull
python scripts/eval_comprehensive.py --existing-results all_experiments_results.json

# 2. 下载结果到本地
# comprehensive_results/comprehensive_results.csv
# comprehensive_results/paper_table_comprehensive.md
# comprehensive_results/confusion_matrices/
# comprehensive_results/training_curves/
# comprehensive_results/per_class_metrics/

# 3. 把 CSV 数值填入本文档 Section 5 的表格
# 4. 根据结果写 Section 6 分析
# 5. 补充定性可视化（去 --no-figures 重跑）
```

---

*最后更新: 2026-08-11 | 16 模型 | 20 指标 | 待填充结果*
