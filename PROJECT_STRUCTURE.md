# Project Structure — ResUNet Enhancement for BraTS2020

> 最后更新: 2026-08-06 | 实验数: 11 | 状态: Phase 1 完成

```
enhance_resu/
│
├── models/                          # 模型架构 (nn.Module only, 不含训练逻辑)
│   ├── __init__.py                  #   导出 21 个符号
│   ├── base_blocks.py               #   DoubleConv, Down, Up, Out (共享组件)
│   ├── unet3d.py                    #   UNet3d 基线
│   ├── resunet3d.py                 #   ResUNet3d 基线 ★ (n_channels=24, 5.8M params)
│   ├── attunet3d.py                 #   AttUNet3d (CBAM + AttentionGate)
│   ├── nnunet3d.py                  #   nnUNet3d (InstanceNorm + LeakyReLU)
│   ├── resunet_edge.py              #   V2: Sobel/Laplacian/Random Edge Branch
│   ├── resunet_fgfe.py              #   V2: FGFE decoder (Yao et al., MICCAI 2025)
│   ├── fgfe_module.py               #   FGFE: LaplacianPyramid3d + cross-attention
│   └── sla_module.py                #   SLA3D: channel/spatial attention (Step 2 预留)
│
├── losses/                          # 损失函数
│   ├── __init__.py                  #   导出 10 个 loss class
│   ├── basics.py                    #   DiceLoss, BCEDiceLoss (基线)
│   └── enhanced.py                  #   CELoss, BoundaryLoss, DiceCEBoundaryLoss (V1)
│                                    #   CCLevelDiceLoss, PMDiceLoss (SLA-FB Step 2)
│                                    #   BCEDiceCCLoss, BCEDicePMLoss, BCEDiceCCPMLoss
│
├── data/                            # 数据管道
│   ├── __init__.py
│   ├── dataset.py                   #   BratsDataset, get_dataloader, BratsDatasetWithFGSampling
│   └── foreground_sampler.py        #   4-strategy patch sampler (STSNet 启发的)
│
├── training/                        # 训练设施
│   ├── __init__.py
│   ├── config.py                    #   GlobalConfig, check_exist, check_exist_last, seed=55
│   ├── trainer.py                   #   Trainer: loop, early-stop(patience=25), checkpoint
│   └── metrics.py                   #   Meter, dice_coef_metric, jaccard_coef_metric
│
├── evaluation/                      # 评估与可视化
│   ├── __init__.py                  #   导出 ~30 个函数
│   ├── evaluator.py                 #   compute_metrics, compute_scores_per_classes, 混淆矩阵
│   ├── visualization.py             #   3D GIF, Plotly 3D, 肿瘤覆盖图
│   ├── advanced_metrics.py          #   Dice(WT/TC/ET), Recall, Precision, HD95, NSD,
│   │                                #   Lesion-wise Recall, Small-case Dice, 边界叠加
│   └── visualize_report.py          #   论文配图: 柱状图 + 定性覆盖 + 小病灶放大
│
├── scripts/                         # 训练 & 评估脚本 (10 个 .py + 1 个 .sh)
│   ├── train_enhanced.py            #   V1: λb=0.1/0.3/0.5 训练入口
│   ├── train_v2_edge.py             #   V2: Edge Branch (sobel/laplacian/random, concat/add)
│   ├── train_fgfe.py                #   V2: FGFE 训练入口
│   ├── train_fg_sampling.py         #   SLA-FB: FG-aware patch sampling
│   ├── train_cc_dice.py             #   SLA-FB: CC-Level Dice Loss
│   ├── train_loss_ablation.py       #   SLA-FB: PM-Dice / CC+PM 统一入口
│   ├── eval_all_experiments.py      #   ★ 统一评估框架 (可扩展注册表)
│   ├── run_all.sh                   #   ★ 顺序训练 runner (智能跳过/续训)
│   ├── et_statistics.py             #   ET 病灶统计分析
│   ├── visualize_lesions.py         #   交互式病灶 3D+2D 验证 (Plotly HTML)
│   └── viz_fg_sampling.py           #   FG sampling 策略可视化验证
│
├── notebooks/                       # Jupyter 实验入口
│   ├── evaluate_baselines.ipynb     #   4 基线模型高级指标对比
│   └── experiment_lambda_results.ipynb  # λb 消融结果对比
│
├── tumourCSV.csv                    # ★ 标准数据划分 (所有实验共用)
│
├── MultiModel XAI Brats2020.ipynb   #   原始 4 基线训练 notebook
├── ResUNet_Enhanced.ipynb           #   ResUNet 增强实验 notebook
│
├── nnunet.py                        #   遗留: nnUNet 独立训练脚本
├── nnunet_model.py                  #   遗留: nnUNet 纯模型定义
├── patch_notebook.py                #   遗留: notebook 补丁工具
│
├── CHANGELOG.md                     #   详细修改日志
├── RESEARCH_RECORDS.md              #   ET 病灶统计研究记录
├── REPORT_ET_LESION_ANALYSIS.md     #   病灶级分析的论文论证
├── COMPREHENSIVE_REPORT.md          #   4 基线模型综合评估报告
├── PROJECT_STRUCTURE.md             #   本文档
│
├── environment.yml                  #   Conda 环境
├── requirements.txt                 #   PyTorch + MONAI + nibabel + ...
├── requirements_clean.txt           #   精简版依赖
├── .gitignore
│
└── brats2020_full_report/           #   归档: 完整旧版项目副本 (不参与当前开发)
```

---

## 核心文件速查 (写论文时最可能用到的)

### 模型定义
| 文件 | 关键 class | 参数量 |
|---|---|---|
| `models/resunet3d.py` | `ResUNet3d` | 5,763,867 |
| `models/resunet_edge.py` | `ResUNetEdge`, `SobelEdge3d`, `LaplacianEdge3d` | +0.3M |
| `models/resunet_fgfe.py` | `ResUNetFGFE` | +0.5M |
| `models/fgfe_module.py` | `FGFE`, `LaplacianPyramid3d` | — |
| `models/sla_module.py` | `SLA3D`, `ChannelAttention3D`, `SpatialAttention3D` | — |

### 损失函数
| 文件 | 关键 class | 公式 |
|---|---|---|
| `losses/basics.py` | `BCEDiceLoss` | L = BCE + Dice (基线) |
| `losses/enhanced.py` | `DiceCEBoundaryLoss` | L = Dice + CE + λb·Boundary (V1) |
| `losses/enhanced.py` | `CCLevelDiceLoss` | per-ET-component Dice, equal weight |
| `losses/enhanced.py` | `PMDiceLoss` | m = |y-p̂|^γ modulated Dice (Hosseini 2025) |

### 训练设施
| 文件 | 关键内容 |
|---|---|
| `training/config.py` | `check_exist()` warm-start, `check_exist_last()` resume, seed=55 |
| `training/trainer.py` | early-stopping (patience=25), ReduceLROnPlateau, train_log.csv |

### 评估指标
| 文件 | 关键函数 |
|---|---|
| `evaluation/advanced_metrics.py` | `hd95_single`, `nsd_single`, `lesion_wise_detection`, `compute_small_case_dice`, `compute_all_advanced_metrics` |
| `evaluation/visualize_report.py` | `generate_all_figures` — 论文全部配图 |

### 入口脚本
| 文件 | 用途 |
|---|---|
| `scripts/run_all.sh` | 一台 GPU 顺序跑完 10 个实验 |
| `scripts/eval_all_experiments.py` | 统一评估 → JSON/CSV/MD/图 |
