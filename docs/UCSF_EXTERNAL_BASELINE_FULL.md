# UCSF-BMSR Baseline/Full 外部数据集训练

这套入口在 UCSF-BMSR v1.3 的461个带标签训练扫描上重新训练二值脑转移瘤分割模型。主实验不使用五折交叉验证，而是延续原 BraTS 实验的一次固定70%/20%/10%训练、验证、测试划分和随机种子10。UCSF包含同一患者的多次扫描，脚本以SubjectID数字部分作为患者组，保证A/B/C等随访扫描不会跨集合泄漏。

当前数据对应314名患者，预期划分为：训练219名患者/321个扫描，验证63名/94个扫描，测试32名/46个扫描。实际计数会写入预处理元数据。

## 公平性约束

Baseline和Full共同使用：

- 相同四通道输入顺序：T1post、T1pre、FLAIR、T1post−T1pre subtraction；该组合来自UCSF官方benchmark。
- 相同二值标签：`<SubjectID>_seg.nii.gz`，不使用只有324例具备的BraTS多区标签。
- 相同预处理缓存、患者划分、样本顺序和验证集。
- 与原BraTS一致的每通道min-max归一化和输入张量尺寸 `(D,H,W)=(100,170,170)`；UCSF原始矩阵和层数不统一，因此先统一到这一尺寸，完整视野不会被标签引导裁剪。
- `n_channels=24`、Adam、学习率 `5e-4`、batch size 1、梯度累积4、最多200 epochs。
- ReduceLROnPlateau patience 2；early stopping patience 25、min_delta `1e-4`。
- 相同主损失 `BCE + global Dice`。

Full固定为正式AFBMS-ResUNet：普通LHFC concat、ABS权重0.1、AR-MSC V2。它从同一seed、同一UCSF划分训练得到的Baseline最佳权重warm-start，与原BraTS Full训练方式一致。模型输出由原来的WT/TC/ET三通道改为一个脑转移瘤通道，这是任务标签变化所必需的唯一输出层适配。

官方benchmark采用nnU-Net v1并允许指定fold或训练`all`，但它没有规定本项目Baseline/Full必须采用五折。五折会改变原论文的训练协议和计算预算，因此本次主外部比较不使用；如以后需要和官方nnU-Net报告方式对齐，可作为独立补充实验。

## AutoDL运行

建议在独立环境安装与PyTorch 2.1兼容的依赖：

```bash
pip install -r requirements_ucsf.txt
```

如果AutoDL镜像已经带有对应CUDA版本的PyTorch，可先删除依赖文件中的
`torch==2.1.2`一行，再安装其余包。不要将PyTorch 2.1与NumPy 2.x组合；
这会使`torch.from_numpy`不可用。

假设数据解压在：

```text
/root/autodl-tmp/UCSF_BrainMetastases_v1.3/UCSF_BrainMetastases_TRAIN
```

先生成固定划分和共享缓存：

```bash
cd /root/mri_deep
python scripts/prepare_ucsf_bmsr.py --dry-run
python scripts/prepare_ucsf_bmsr.py
```

若路径不同：

```bash
python scripts/prepare_ucsf_bmsr.py \
  --data-root /你的路径/UCSF_BrainMetastases_TRAIN \
  --output-root /root/autodl-tmp/ucsf_bmsr_prepared
```

检查协议而不启动训练：

```bash
python scripts/train_ucsf_baseline_full.py --dry-run
```

顺序训练Baseline和Full。Baseline完成标记和best checkpoint存在后，Full才会启动：

```bash
python scripts/train_ucsf_baseline_full.py
```

中断后重复同一命令会从各自目录的last checkpoint恢复；为控制数据盘占用，
每个模型只保留最新的恢复checkpoint和验证集最佳checkpoint，已经完成的模型默认跳过。只继续Full：

```bash
python scripts/train_ucsf_baseline_full.py --models full
```

测试集统一评价：

```bash
python scripts/eval_ucsf_baseline_full.py
```

默认输出：

```text
/root/autodl-tmp/ucsf_baseline_full/seed55/
├── baseline/
├── full/
└── evaluation/
    ├── summary.csv
    ├── per_case.csv
    ├── per_gt_lesion.csv
    └── paired_case_dice.csv
```

评价默认沿用原项目概率阈值0.33、26连通、一对一最大Dice匹配、预测和GT组件最小10体素。主结果`summary.csv`只保留五项与BraTS最终分析对应的指标：整体Dice、HD95、病灶级Recall、全部真实病灶漏检记0的GT-anchored Dice，以及小病灶漏检记0的GT-anchored Dice。UCSF是单类二值任务，因此Macro Dice与ET Dice合并为同一个整体Dice，不重复报告。小/中/大病灶仍采用原实验的“训练集拟合、测试集冻结”原则，但考虑UCSF扫描的物理间距不统一，主划分依据单个GT连通病灶的物理体积（mm³），而不复用BraTS的固定体素界值。脚本会从固定训练集拟合近似三等分界值并输出`training_lesion_size_distribution.csv`和`summary_by_lesion_size.csv`。测试集阈值不得根据最终结果再调；如需调概率阈值，应只在固定验证集完成，然后把选定阈值用于一次测试。

## 运行前检查

安装仓库依赖后运行：

```bash
python -m pytest tests/test_ucsf_external_training.py -q
```

预处理会生成压缩NPZ，所需磁盘空间取决于影像内容，建议先确认 `/root/autodl-tmp` 有足够余量。原始UCSF数据及官方benchmark目录受数据许可约束，均被 `.gitignore` 排除，不会上传GitHub。
