# ResUNet 增强 — BraTS2020 脑肿瘤分割

在 **ResUNet3d** 基线上，从 **损失 / 架构 / 数据采样** 三个维度做单变量增强，目标是改善 **ET（增强肿瘤）小病灶**的边界与检出。

## 目录结构

```
enhance_resu/
├── models/       模型（只定义网络结构，不含训练逻辑）
│   ├── base_blocks.py          DoubleConv / Down / Up / Out（共享组件）
│   ├── resunet3d.py            ★ 基线 ResUNet3d
│   ├── unet3d.py               UNet3d（基线对照）
│   ├── attunet3d.py            AttUNet3d（CBAM + 注意力门控）
│   ├── nnunet3d.py             nnUNet3d（InstanceNorm + LeakyReLU）
│   ├── resunet_edge.py         V2 边缘分支（Sobel/Laplacian，concat/add）
│   ├── resunet_hf_boundary.py  V2 HF 边界双头
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

**ResUNetFGFE** — 解码器 `ResUp` 换成 `ResUpFGFE`：Laplacian 分解特征为高/低频 → 交叉注意力 → 残差。

**SLA3D** — 通道 + 空间注意力，设计用于解码器高分辨率层，当前预留未接入训练。

**基线对照**：

| 模型 | 相对 ResUNet3d |
|---|---|
| UNet3d | 无残差（DoubleConv） |
| AttUNet3d | CBAM + 注意力门控 |
| nnUNet3d | InstanceNorm + LeakyReLU + 步长卷积 |

> 4 个基线模型代码与原始 notebook `MultiModel XAI Brats2020.ipynb` 逐层一致，仅做了模块化拆分。

## 损失函数

| 损失 | 作用 |
|---|---|
| `BCEDiceLoss` | BCE + Global Dice（基线） |
| `DiceCEBoundaryLoss` | + 距离边界损失（V1） |
| `CCLevelDiceLoss` | ET 连通域级 Dice（病灶级） |
| `PMDiceLoss` | 像素调制 Dice |
| `BCEDiceWithBoundaryLoss` | BCEDice + 边界 BCE（HF 双头） |

## 快速上手

```bash
pip install -r requirements_clean.txt   # PyTorch 2.1.2 + MONAI + nibabel

# 训练
python scripts/train_hf_boundary.py --edge_type laplacian --boundary_weight 0.2

# 评估
python scripts/eval_all_experiments.py
```

> `training/config.py` 中数据 / checkpoint 路径硬编码为 AutoDL `/root/autodl-tmp/...`，换环境需自行修改。
