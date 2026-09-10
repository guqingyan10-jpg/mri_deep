# ResUNet 增强 — BraTS2020 脑肿瘤分割

在 **ResUNet3d** 基线上，从 **损失 / 架构 / 数据采样** 三个维度做单变量增强，目标是改善 **ET（增强肿瘤）小病灶**的边界与检出。

> 新增评价代码审核入口：**[CODE_REVIEW_GUIDE.md](CODE_REVIEW_GUIDE.md)**

## 目录结构

```
enhance_resu/
├── models/       模型（只定义网络结构，不含训练逻辑）
│   ├── base_blocks.py          DoubleConv / Down / Up / Out（共享组件）
│   ├── resunet3d.py            ★ 基线 ResUNet3d
│   ├── unet3d.py               UNet3d（基线对照）
│   ├── attunet3d.py            AttUNet3d（CBAM + 注意力门控）
│   ├── nnunet3d.py             旧版 nnU-Net-inspired surrogate（仅兼容旧权重）
│   ├── nnunet_plainconv3d.py   nnU-Net v2 PlainConvUNet（统一训练协议）
│   ├── resunet_edge.py         V2 边缘分支（Sobel/Laplacian，concat/add）
│   ├── resunet_hf_boundary.py  V2 HF 边界双头
│   ├── resunet_hf_concat_boundary.py  最终组合：多尺度 Laplacian concat + 边界双头
│   ├── resunet_fgfe.py         V2 频域增强
│   ├── fgfe_module.py          FGFE / LaplacianPyramid3d
│   └── sla_module.py           SLA3D 小病灶注意力（预留）
├── losses/       损失（basics.py 基线 / enhanced.py 增强）
├── data/         dataset.py 数据加载 + foreground_sampler.py patch 采样
├── training/     Trainer / config / metrics
├── evaluation/   HD95、NSD、病灶级等指标 + 配图
├── scripts/      训练 & 评估入口
├── tumourCSV.csv 标准数据划分（所有实验共用）
└── PROJECT_STRUCTURE.md  完整文件树与核心代码速查
```

> 每个文件的详细说明见 **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**。

## 模型改动

所有增强模型以 `ResUNet3d`（`n_channels=24`）为基线，**每次只改一个模块**。

**ResUNetEdge** — 新增多尺度边缘分支：`Sobel/LaplacianEdge3d` 对原始 MRI 提取边缘 → `EdgePyramid` 多尺度化 → 在解码器每层用 `ResUpEdge` 做 concat / add 注入。

**ResUNetHFBoundary** — 解码器最后一层注入固定边缘特征 + 边界预测双头：

```
dec4_out (B,24,128³) + hf_aligned(1×1 对齐的 Sobel/Laplacian) → fused
fused → seg_head(1×1) → seg          fused → boundary_head → boundary
```

配合 `BCEDiceWithBoundaryLoss`（主分割 + λ·边界 BCE）。

**ResUNetHFConcatBoundary** — 最终组合模型：复用 `ResUNetEdge` 的 Laplacian
高频残差与四级 `EdgePyramid`，在 `dec1`–`dec4` 逐层 concat，并增加与 HF
Boundary 相同的边界辅助头。训练损失固定为
`BCEDiceLoss(seg, GT) + 0.3 × BCE(boundary, boundary_GT)`。

**ResUNetFGFE** — 解码器 `ResUp` 换成 `ResUpFGFE`：Laplacian 分解特征为高/低频 → 交叉注意力 → 残差。

**SLA3D** — 通道 + 空间注意力，设计用于解码器高分辨率层，当前预留未接入训练。

**基线对照**：

| 模型 | 相对 ResUNet3d |
|---|---|
| UNet3d | 无残差（DoubleConv） |
| AttUNet3d | CBAM + 注意力门控 |
| Legacy nnUNet3d surrogate | InstanceNorm + LeakyReLU + 步长卷积；不是官方 nnU-Net |
| nnU-Net v2 PlainConvUNet | 官方架构后端；使用本工程统一训练协议 |

> 原有 4 个基线模型代码与 notebook 一致；其中旧 `nnUNet3d` 仅为历史权重兼容，不再作为论文强基线。

新的 nnU-Net 对照直接使用官方 `dynamic-network-architectures` 提供的
`PlainConvUNet`，但与 U-Net/ResUNet 一样使用当前工程的输入、BCE-Dice、
Adam (`5e-4`)、200 epochs 设置及固定 257/74/37 划分。运行方法见
[`docs/NNUNET_PLAINCONV_UNIFIED_BASELINE.md`](docs/NNUNET_PLAINCONV_UNIFIED_BASELINE.md)。

## 损失函数

基线为 `BCEDiceLoss = BCE + Global Dice`。

**损失维度**（5 组单变量消融）：

| 实验 | class | 公式 | 相对基线 |
|---|---|---|---|
| CC-Dice | `BCEDiceCCLoss` | BCE + Global Dice + λ_cc·CC-Dice | 基线 **+ CC-Dice** |
| PM-Dice (γ=2) | `BCEDicePMLoss` | BCE + Global Dice + λ_pm·PM-Dice | 基线 **+ PM-Dice** |
| BCE + CC-Dice | `BCECCDiceLoss` | BCE + λ_cc·CC-Dice | ⚠ **无 Global Dice**（CC 替换） |
| BCE + PM-Dice | `BCEPMDiceLoss` | BCE + λ_pm·PM-Dice | ⚠ **无 Global Dice**（PM 替换） |
| Full | `BCEDiceCCPMLoss` | BCE + Global + λ_cc·CC + λ_pm·PM | 全组合 |

命名规律（最易混的一对）：**`BCEDice*`** = 保留 Global Dice，在基线上**加**新项；**`BCE*`** = **去掉** Global Dice，用新项**替换**。

**其余维度**：

| 实验 | class | 公式 |
|---|---|---|
| V1 边界 | `DiceCEBoundaryLoss` | Dice + CE + λb·Boundary |
| HF 双头 | `BCEDiceWithBoundaryLoss` | BCEDiceLoss(seg) + λ·BCE(boundary) |

> `DiceLoss` / `CELoss` / `BoundaryLoss` / `CCLevelDiceLoss` / `PMDiceLoss` 是上述组合内部的组件，不单独训练。

## 快速上手

```bash
pip install -r requirements_clean.txt   # PyTorch 2.1.2 + MONAI + nibabel

# 训练
python scripts/train_hf_boundary.py --edge_type laplacian --boundary_weight 0.2

# 最终组合模型（Laplacian 多尺度 concat + 0.3 边界辅助监督）
python scripts/train_hf_concat_boundary.py

# 评估
python scripts/eval_all_experiments.py
```

> `training/config.py` 中数据 / checkpoint 路径硬编码为 AutoDL `/root/autodl-tmp/...`，换环境需自行修改。

## UCSF-BMSR 外部数据集复训

UCSF-BMSR上的二值脑转移瘤Baseline/Full配对训练使用固定患者分组的
70%/20%/10%划分，不采用五折。准备数据、训练、恢复和统一评价命令见
[`docs/UCSF_EXTERNAL_BASELINE_FULL.md`](docs/UCSF_EXTERNAL_BASELINE_FULL.md)。

## Seed123 门控配对实验

门控实验只改变四级 Laplacian 特征进入 decoder concat 前的融合：

```text
edge feature -> (1 + tanh(gate)) * edge feature -> concat
```

第一组保持原始 `BCEDiceLoss` 且不增加 Boundary Head；第二组保留现有
HF Concat Boundary 的双头结构和 `boundary_weight=0.1`。两组都从
`/root/autodl-tmp/stability/seed123/baseline/best_model_*.pth` warm-start，
并保持 `n_channels=24`、学习率 `5e-4`、最多 200 epochs、梯度累积 4、
早停 patience 25 和原数据划分不变。

```bash
# 先检查将要执行的两条命令
python scripts/run_gated_seed_screen.py --seed 123 --dry_run

# 顺序训练 Edge gated 和 HF gated Boundary w=0.1
python scripts/run_gated_seed_screen.py --seed 123

# 训练完成后，用四个核心指标评估 seed123 的现有与门控模型
python scripts/eval_key_comparison.py --seed 123 --no-timing --no-cache
```

新增 checkpoint 目录：

```text
/root/autodl-tmp/stability/seed123/edge_laplacian_gated_concat
/root/autodl-tmp/stability/seed123/hf_gated_concat_boundary_w0.1
```

核心指标固定为 Macro Dice、ET Dice、ET HD95 和 Small-case ET Dice。

## Multi-scale context 配对实验（seed 42 / 123）

该实验仅在现有 `Laplacian EdgePyramid + concat + Boundary w=0.1`
模型的编码器瓶颈增加 `MultiScaleContext3d`。数据划分、完整 `128³`
输入、基础通道数 24、学习率 `5e-4`、最多 200 epochs、梯度累积 4、
早停 patience 25，以及每个 seed 的 baseline warm-start 均保持不变。

上传代码到 AutoDL 后先运行轻量 smoke test：

```bash
python -m unittest \
  tests.test_multiscale_context_integration.MultiScaleContextIntegrationTests.test_enabled_model_preserves_shape_and_shared_initialization \
  -v
```

确认两个 seed 的 baseline 已完成后，检查并执行训练命令：

```bash
python scripts/run_multiscale_seed_screen.py --seeds 42 123 --dry_run
python scripts/run_multiscale_seed_screen.py --seeds 42 123
```

训练完成后分别与同 seed 的现有对照模型评估：

```bash
python scripts/eval_key_comparison.py --seed 42 --no-timing --no-cache
python scripts/eval_key_comparison.py --seed 123 --no-timing --no-cache
```

新增 checkpoint 目录：

```text
/root/autodl-tmp/stability/seed42/hf_concat_boundary_w0.1_multiscale
/root/autodl-tmp/stability/seed123/hf_concat_boundary_w0.1_multiscale
```

### Multi-scale context V2（identity-start，先试 seed 42）

V2 保留四个多尺度分支，但使用 `x + alpha * context`，其中 `alpha`
初始为 0；因此从 baseline warm-start 后，训练第一步仍与原模型一致。
V2 使用独立目录，不覆盖 V1：

```bash
python scripts/run_multiscale_v2_seed_screen.py --seeds 42 --dry_run
python scripts/run_multiscale_v2_seed_screen.py --seeds 42
python scripts/eval_key_comparison.py --seed 42 --no-timing --no-cache
```

输出目录：

```text
/root/autodl-tmp/stability/seed42/hf_concat_boundary_w0.1_multiscale_v2
```

### V2 alpha 最小敏感性

该诊断固定三个种子的 V2 `best_model` 和37例测试集，只在推理时将标量
`alpha` 分别设置为 0、checkpoint 学习值和 1。seed55 使用主实验目录，
seed42/123 使用 stability runner 目录；脚本会在结果中保留训练协议标签，
因此三种子的汇总仅作描述性报告。

三项主指标为 Macro Dice、ET Dice 和病灶级
Small-lesion ET GT-anchored Dice。Small病灶使用训练集拟合并冻结的
`et_training_lesion_strata.json`，复用现有26邻域、一对一匹配和最小10
体素口径；漏检的small GT病灶Dice记为0。脚本同时输出small病灶的
Matched Dice、Recall、Miss rate和逐病灶明细。

```bash
python scripts/eval_alpha_sensitivity.py
```

正式运行前可先快速核对三个 best checkpoint 路径和其中的学习 alpha：

```bash
python scripts/eval_alpha_sensitivity.py --inspect-only
```

默认 checkpoint 目录：

```text
seed42:  /root/autodl-tmp/stability/seed42/hf_concat_boundary_w0.1_multiscale_v2
seed55:  /root/autodl-tmp/ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model
seed123: /root/autodl-tmp/stability/seed123/hf_concat_boundary_w0.1_multiscale_v2
```

若实际目录不同，可重复覆盖：

```bash
python scripts/eval_alpha_sensitivity.py \
  --checkpoint-dir 42=/path/to/seed42_v2 \
  --checkpoint-dir 55=/path/to/seed55_v2 \
  --checkpoint-dir 123=/path/to/seed123_v2
```

输出保存在 `alpha_sensitivity_test_results/`，包括逐种子结果、逐病例 Dice、
逐ET病灶明细、small病灶固定清单、best-checkpoint alpha 均值/范围和
敏感性图。

### seed123 alpha 学习轨迹补充训练

下面的独立运行保持原 seed123 V2 目录不变，复用同一个 seed123 baseline
best checkpoint，并写入新的 `_alpha_trace` 目录：

```bash
python scripts/run_seed123_alpha_trace.py --dry-run
python scripts/run_seed123_alpha_trace.py
```

新目录：

```text
/root/autodl-tmp/stability/seed123/hf_concat_boundary_w0.1_multiscale_v2_alpha_trace
```

其中 `alpha_history.csv` 保存初始化及每个 epoch 的 alpha、学习率和损失；
`alpha_learning_curve.png/.pdf` 绘制轨迹并标记 best epoch。使用该次重训的
best checkpoint 做 seed123 敏感性分析时显式覆盖目录：

```bash
python scripts/eval_alpha_sensitivity.py \
  --checkpoint-dir 123=/root/autodl-tmp/stability/seed123/hf_concat_boundary_w0.1_multiscale_v2_alpha_trace
```

### 三种子典型病灶与边界病例筛选

正式筛选同时运行 seed42、seed55、seed123，并固定使用同一批37例测试集。
每个 seed 均输出病灶可视化图；脚本还会按同一个GT小病灶或同一个病例
汇总跨 seed 一致性，优先推荐三个 seed 均改善的候选，避免只挑某个 seed
的最佳结果。seed55 保留主实验路径，seed42/123 使用 stability runner 路径。

```bash
python scripts/run_multiseed_typical_case_analysis.py --dry-run
python scripts/run_multiseed_typical_case_analysis.py
```

默认会跳过结果文件齐全的 seed，因此中断后可直接重跑；只有需要强制重算
全部三个 seed 时才加 `--rerun-existing`。若某个 seed 没有正向的匹配小病灶
Dice 增益，脚本会保留完整统计并将该 seed 的图明确标为“最佳可比较、非正向
改善”，不会因缺少理想病例而中止或误报。

每个 seed 的图位于
`boundary_typical_case_multiseed_results/seed{42,55,123}/`；跨 seed 排名为
`cross_seed_small_lesion_ranking.csv` 和
`cross_seed_boundary_case_ranking.csv`。

### 交接工程审核材料

在 AutoDL 上统一收集指定基线、LHFC/ABS/MSC/AR-MSC 模型的正式 best
checkpoint、训练记录和结果数据：

```bash
python scripts/stage_handoff_review.py
```

输出目录为 `handoff_review_data/`。模型统一命名及源目录映射记录在
`manifests/model_manifest.csv`；正式、探索性 valid+test 和旧结果分别存放。
脚本不收集成图、绘图脚本、原始 BraTS 影像、last-epoch 权重或缓存。
