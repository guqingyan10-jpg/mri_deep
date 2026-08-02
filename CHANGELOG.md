# Project Changelog — ResUNet Enhancement for BraTS2020

> 记录所有修改、新增模块、实验配置，便于论文写作和模型架构图绘制。

---

## 2026-08-01 | 项目重构：模块化拆分

### 目的
将 `MultiModel XAI Brats2020.ipynb` 中的代码按功能拆分为独立模块，消除 `nnunet.py`、`resunet_enhanced.py` 之间的代码重复，便于后续改进实验。

### 新建目录结构
```
enhance_resu/
├── models/           # 模型架构（纯 nn.Module，不含训练逻辑）
├── losses/           # 损失函数（basics.py + enhanced.py）
├── data/             # 数据管道（BratsDataset, get_dataloader）
├── training/         # 训练设施（Trainer, metrics, config）
├── evaluation/       # 评估与可视化（evaluator, visualization）
├── scripts/          # 辅助脚本（统计、分析等）
└── notebooks/        # （待建）Jupyter 实验入口
```

### 模块来源映射

| 新文件 | 原始来源 |
|---|---|
| `models/base_blocks.py` | Notebook cell 42 (DoubleConv, Down, Up, Out) |
| `models/unet3d.py` | Notebook cell 42 (UNet3d) |
| `models/resunet3d.py` | Notebook cell 46 (ResBlock, ResDown, ResUp, FirstLayer, ResUNet3d) |
| `models/attunet3d.py` | Notebook cells 51+53 (CBAM, attention_gate, AttUp, AttUNet3d) |
| `models/nnunet3d.py` | `nnunet_model.py` (nnDoubleConv, nnDown, nnUp, nnUNet3d) |
| `losses/basics.py` | Notebook cell 37 (DiceLoss, BCEDiceLoss) |
| `losses/enhanced.py` | `resunet_enhanced.py` (CELoss, BoundaryLoss, DiceCEBoundaryLoss) |
| `data/dataset.py` | Notebook cell 27 (BratsDataset, get_dataloader) |
| `training/trainer.py` | Notebook cell 39 (Trainer class) |
| `training/metrics.py` | Notebook cell 37 (Meter, dice/jaccard metrics) |
| `training/config.py` | Notebook cells 10+15 (GlobalConfig, seed_everything, check_exist) |
| `evaluation/evaluator.py` | Notebook cells 70,71,72,84,98,116 |
| `evaluation/visualization.py` | Notebook cells 8,117,118 |

### 原有文件：未修改
- `MultiModel XAI Brats2020.ipynb`
- `ResUNet_Enhanced.ipynb`
- `resunet_enhanced.py`
- `nnunet.py`
- `nnunet_model.py`
- `patch_notebook.py`

---

## 基线模型架构速查

| 组件 | UNet3d | ResUNet3d | AttUNet3d | nnUNet3d |
|---|---|---|---|---|
| 卷积块 | DoubleConv | ResBlock | DoubleConv | nnDoubleConv |
| 归一化 | GroupNorm | GroupNorm | GroupNorm | InstanceNorm |
| 激活函数 | ReLU | ReLU | ReLU | LeakyReLU |
| 下采样 | MaxPool3d | MaxPool3d | MaxPool3d | Strided Conv3d |
| 上采样 | Trilinear | Trilinear | Trilinear | ConvTranspose3d |
| Skip连接 | Concat | Concat+残差 | Concat+CBAM+AG | Concat |
| 参数量(~n_ch=24) | ~7.8M | ~8.1M | ~8.5M | ~8.35M |

---

## 损失函数速查

| 损失函数 | 公式 | 来源文件 |
|---|---|---|
| DiceLoss | L = 1 - 2|P∩T|/(|P|+|T|) | `losses/basics.py` |
| BCEDiceLoss | L = BCE + Dice | `losses/basics.py` (基线用) |
| CELoss | 加权 BCE (WT:1, TC:2, ET:4) | `losses/enhanced.py` |
| BoundaryLoss | Laplacian边缘加权BCE | `losses/enhanced.py` |
| DiceCEBoundaryLoss | α·Dice + β·CE + γ·Boundary | `losses/enhanced.py` |

---

## 实验公平性保证

- 数据划分：`tumourCSV.csv` + `train_test_split(random_state=10, test_size=0.3)`
- 所有模型相同超参：lr=5e-4, Adam, ReduceLROnPlateau(patience=2), batch_size=1, accumulation=4, n_channels=24
- 数据预处理：`data[40:210, 40:210, 20:120]` crop + min-max normalize

---

---

## 2026-08-01 | ET 统计分析脚本

### 新增 `scripts/et_statistics.py`
对 369 例 BraTS2020 训练集逐例统计：

**体积指标:**
- ET / WT / TC 体素数（label 4 / labels 1+2+4 / labels 1+4）
- ET/WT 占比、ET/TC 占比

**多灶性分析（Connected Component Analysis）:**
- 方法: `scipy.ndimage.label()` 对 ET mask 做 3D 26-邻接连通域标记
- 过滤: 排除 < 10 体素的碎片（标注噪声）
- 输出:
  - 每个病例的 ET 病灶数、多灶性标志
  - 最大/最小病灶体素、主病灶占比
  - 每个独立病灶的详细记录（et_components_detail.csv）

**输出文件:**
- `et_statistics.csv` — 病例级统计
- `et_components_detail.csv` — 病灶级明细

### 本地数据问题
Windows 环境 nibabel 加载大 nii 文件时内存不足，需在服务器上运行。

---

---

## 2026-08-01 | 新增高级评估指标模块

### 新增 `evaluation/advanced_metrics.py`

| 指标 | 函数 | 说明 |
|---|---|---|
| **Per-class Recall** | `per_class_recall_precision()` | TP/(TP+FN) — 真实肿瘤中被检出的比例 |
| **Per-class Precision** | `per_class_recall_precision()` | TP/(TP+FP) — 预测为肿瘤中正确的比例 |
| **ET/TC HD95** | `hd95_single()` | 95th percentile Hausdorff Distance (mm)，边界距离误差 |
| **Lesion-wise Recall** | `lesion_wise_detection()` | 独立ET病灶的检出率（连通域级别，非像素级） |
| **Lesion-wise Precision** | `lesion_wise_detection()` | 预测病灶中有多少对应真实病灶 |
| **Small-case Dice** | `compute_small_case_dice()` | 仅在小ET病例（bottom 25%）上计算的Dice |
| **Boundary Overlay** | `boundary_overlay()` / `save_boundary_comparison()` | GT vs Pred边界轮廓叠加可视化 |

**HD95 实现:** 使用 `scipy.ndimage.distance_transform_edt`（向量化距离变换），无需 medpy 等额外依赖。对称计算：max(HD95(GT→Pred), HD95(Pred→GT))。

**Lesion-wise Recall 实现:** `scipy.ndimage.label()` 对 GT ET mask 做 3D 连通域标记 → 对每个 GT 病灶检查是否有 Pred 病灶重叠 → 检出数 / 总病灶数。

### 新增 `notebooks/evaluate_baselines.ipynb`
- 10 个 cell，一键评估 4 个基线模型
- 可独立运行，也可在原 notebook 训练完后接着跑
- 自动生成对比表 + 4 张分析图 + 边界可视化

### 使用方式
```python
# 在原 notebook 末尾加两行：
from evaluation.advanced_metrics import compute_all_advanced_metrics, print_comparison_table

unet_m = compute_all_advanced_metrics(UNet, test_dataloader, model_name='UNet')
resunet_m = compute_all_advanced_metrics(ResUNet, test_dataloader, model_name='ResUNet')
# ...
print_comparison_table([unet_m, resunet_m, attunet_m, nnunet_m])
```

---

## 2026-08-01 | λb 调优实验：Dice+CE+Boundary Loss

### 实验设计
**Loss 公式:** `L = 1.0 × DiceLoss + 0.5 × CELoss + λb × BoundaryLoss`

| 参数 | 值 | 说明 |
|---|---|---|
| α (Dice) | 1.0 | 固定 |
| β (CE) | 0.5 | 固定 |
| **λb (Boundary)** | **0.1, 0.3, 0.5** | 调优变量 |
| Class weights | WT=1.0, TC=3.0, ET=5.0 | ET/TC 高权重 |

### 修改 `losses/enhanced.py`
- 默认 class_weights: `[1.0, 2.0, 4.0]` → `[1.0, 3.0, 5.0]`

### 新增 `scripts/train_enhanced.py`
- 命令行训练脚本
- 用法: `python scripts/train_enhanced.py --lambda_b 0.3`
- 自动从 ResUNet baseline checkpoint warm-start
- 每个 λb 保存到独立目录 `/root/autodl-tmp/ResUNet_Enhanced_lb{λb}_model/`

### 新增 `notebooks/experiment_lambda_results.ipynb`
- 加载 baseline + 3 个 λb 模型
- 对比 ET/TC Dice、ET/TC HD95、Lesion Recall、边界可视化
- 自动选出最优 λb
- 输出 `lambda_experiment_results.csv`（论文表格用）

### 服务器运行步骤
```bash
cd /root/autodl-tmp/mri_deep && git pull

# 依次训练 3 个 λb (每个 ~200 epochs):
python scripts/train_enhanced.py --lambda_b 0.1
python scripts/train_enhanced.py --lambda_b 0.3
python scripts/train_enhanced.py --lambda_b 0.5

# 训练完后，打开 notebooks/experiment_lambda_results.ipynb 评估对比
```

---

## 2026-08-02 | 训练规范：tmux 持久化 + 早停标准

### 标准训练流程（每次训练必须遵守）

```bash
# ===== 第 1 步：创建 tmux 会话（断网/VSCode关闭也不停） =====
tmux new -s train_lb03    # 自己命名，如 train_lb03, train_lb05, train_v2

# ===== 第 2 步：更新代码并开始训练 =====
cd /root/autodl-tmp/mri_deep && git pull
python scripts/train_enhanced.py --lambda_b 0.3

# ===== 第 3 步：退出 tmux（训练继续跑） =====
# 按 Ctrl+B 然后按 D

# ===== 第 4 步：关 VS Code / 断网 / 关电脑，训练不受影响 =====

# ===== 下次连上后查看状态 =====
tmux ls                      # 列出所有会话
tmux attach -t train_lb03    # 进入查看训练进度
# Ctrl+C                      # 在 tmux 内停止训练
```

### tmux 速查表

| 操作 | 命令 |
|---|---|
| 创建会话 | `tmux new -s 名字` |
| 退出（不停） | `Ctrl+B` 然后 `D` |
| 重新进入 | `tmux attach -t 名字` |
| 查看所有 | `tmux ls` |
| 停止训练 | 在会话内 `Ctrl+C` |
| 删除会话 | `tmux kill-session -t 名字` |

### 早停标准（所有实验统一）

| 参数 | 值 | 说明 |
|---|---|---|
| `early_stopping_patience` | 25 | 连续 25 epoch val_loss 不降 → 停止 |
| `min_delta` | 1e-4 | val_loss 下降不足 1e-4 → 不算改善 |
| 监控指标 | `val_loss` | 每次验证后比较 |
| 选择 checkpoint | 最低 val_loss 的 epoch | 与 baseline 选择标准一致 |

**为什么用这些参数:** warm-start 从 pretrained baseline 出发，通常在 10-30 epoch 收敛。25 epoch 耐心值在收敛后给足够余量，同时避免浪费 GPU。

---

## 2026-08-02 | V1 λb 消融实验结果

### 实验配置
- 架构: ResUNet3d (n_channels=24)
- 训练数据: BraTS2020 Training Set (70%)
- 评估数据: BraTS2020 Test Set (独立划分)
- Warm-start: ResUNet baseline (BCEDiceLoss, epoch 199)
- 早停: patience=25, min_delta=1e-4
- Loss: `L = 1.0 × Dice + 0.5 × CE + λb × Boundary`
  - Class weights: WT=1.0, TC=3.0, ET=5.0
  - Boundary: Kervadec 2019 distance-transform BD Loss

### 结果

| 指标 | Baseline (BCEDice) | λb=0.1 | λb=0.3 | 最优 |
|---|---|---|---|---|
| ET Dice | 0.7585 | 0.7534 | **0.7665** | λb=0.3 |
| ET Recall | 0.7775 | 0.7951 | **0.7934** | λb=0.1 |
| ET Precision | 0.7825 | 0.7442 | **0.7672** | Baseline |
| ET HD95 (mm) | **10.26** | 12.30 | 11.77 | Baseline (越低越好) |
| TC HD95 (mm) | **9.02** | 10.41 | 9.31 | Baseline (越低越好) |
| Lesion Recall | 0.718 | **0.749** | 0.741 | λb=0.1 |
| Small-case Dice | 0.621 | 0.634 | **0.642** | λb=0.3 |

### 关键发现

1. **λb=0.3 在分割指标上最优** — ET Dice +0.8%, Small-case Dice +2.1%
2. **检出率提升但边界退化** — Lesion Recall +2.3% 但 ET HD95 +1.5mm
3. **Pure loss-level boundary supervision is insufficient** — 需要 V2 (高频边缘特征)
4. **λb=0.1 过弱** — Dice 反而不如 baseline
5. λb=0.5 训练未完成，待补充

### 论文表述
> Adding the Kervadec 2019 boundary distance loss (λb=0.3) improved ET Dice by 0.8% and Lesion-wise Recall by 2.3%. However, ET HD95 increased from 10.26mm to 11.77mm, suggesting loss-level boundary supervision alone is insufficient for boundary quality. This motivates explicit high-frequency edge feature integration (V2).

---

### 📝 术语解释：Baseline + ET/TC Patch 采样

讲义中提到的 Exp-3 "baseline + ET/TC patch 采样":

**问题:** BraTS2020 中 ET 体素仅占全图的 ~0.5%。随机采样 patch 时，ET 区域几乎不会出现 → 模型缺乏足够的 ET 训练信号。

**方法:** 在训练时以更高概率从包含 ET/TC 的区域采样 3D patch：
- 统计每个病例中 ET/TC 的质心位置
- 采样 patch 时，50% 概率以 ET/TC 质心为中心
- 另外 50% 概率随机采样
- 保证小病灶在训练中出现的频率大幅提升

**与你当前方法的关系:** 你用的 class_weights (ET=5, TC=3, WT=1) 和 patch 采样是**同一目标的两种实现**——都在让小病灶获得更多梯度。class weights 在 loss 层面，patch 采样在数据层面。两者可以组合使用。

---

*最后更新: 2026-08-02*
