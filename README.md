# AFBMS-ResUNet

用于多模态脑肿瘤 MRI 分割的 3D ResUNet 研究代码。论文主模型由四级局部高频特征融合（LHFC）、训练期边界辅助监督（ABS，权重 **0.1**）和瓶颈处的自适应残差多尺度上下文（AR-MSC V2）组成。基础通道数为 24；BraTS2020 输出 WT、TC、ET，UCSF-BMSR 复训为单通道脑转移瘤分割。不要把较早的 `w=0.3` 实验或 V1/V3 上下文模块当作论文主模型。

## 仓库内容

| 目录 | 用途 |
| --- | --- |
| `models/`, `losses/`, `training/`, `data/` | 网络、损失、训练及数据读取；论文主模型在 `models/resunet_hf_concat_boundary.py` |
| `scripts/` | BraTS 和 UCSF 的训练、评估、病灶分析及图像示例生成入口 |
| `evaluation/` | 区域、边界与病灶级评价实现 |
| `tests/` | 模型连接、数据划分、匹配规则及脚本契约测试 |
| `docs/` | 外部数据复训和对照实验的专项说明 |
| `tumourCSV.csv` | BraTS2020 病例清单和本项目固定划分使用的源文件 |

旧 Notebook、内部计划稿及本机导出的依赖清单已从当前代码树移除；需要核对历史时仍可通过 Git 提交历史查看。

## 数据与环境

论文在 BraTS2020 **有标签训练集的 368 例**上固定划分为 257/74/37 例；这 37 例是本项目留出的测试集，**不是 BraTS 官方隐藏测试集**。BraTS 原始数据与引用要求见 [CBICA 官方页面](https://www.med.upenn.edu/cbica/brats2020/data.html)；本项目所用副本来自 [Kaggle 镜像](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)。UCSF-BMSR 的 461 个公开带标签扫描按患者分组后用于独立数据集复训和测试；访问与许可见 [UCSF 数据页面](https://imagingdatasets.ucsf.edu/dataset/1)。请勿将原始 MRI 或权重提交到本仓库。

建议 Python 3.10。先建立隔离环境，再运行 `pip install -r requirements.txt`；已有匹配 CUDA 的 PyTorch 时，可保留该轮子并安装其余依赖。官方 nnU-Net v2 PlainConv 架构对照另需 `requirements_nnunet_architecture.txt`。UCSF 专项环境清单保留在 `requirements_ucsf.txt`。

`tumourCSV.csv` 中的 `path` 列保存原 AutoDL 数据位置。换机器后须将其映射为当地的 BraTS 图像目录；不要修改病例 ID、随机划分或测试集组成来适配路径。

## 论文主模型

先得到同一随机种子的 ResUNet BCE–Dice 基线最佳权重，再从该权重初始化主模型。下例以 seed 55 为例，路径需按本机实际位置填写：

```bash
python scripts/train_baseline_bcedice.py \
  --seed 55 --checkpoint_dir /path/to/seed55/baseline

python scripts/train_hf_concat_boundary.py \
  --seed 55 --fusion concat --boundary_weight 0.1 \
  --multiscale_context_v2 \
  --baseline_checkpoint /path/to/seed55/baseline/best_model_N.pth \
  --checkpoint_dir /path/to/seed55/full
```

对 seed 42/123 的配对运行可先检查：

```bash
python scripts/run_multiscale_v2_seed_screen.py --seeds 42 123 --dry_run
```

基线、消融和病灶级评价入口分别见 `scripts/eval_key_comparison.py`、`scripts/eval_et_lesion_stratified.py` 与 `scripts/eval_et_small_lesion_five_models.py`。α 敏感性分析使用 `scripts/eval_alpha_sensitivity.py`。这些脚本有部分 AutoDL 默认路径；在新机器运行前先检查脚本参数和权重位置。病灶匹配与 GT-anchored Dice 的核对入口见 [代码审核索引](CODE_REVIEW_GUIDE.md)。

UCSF-BMSR 的患者级划分、Baseline/Full 复训和统一评价见 [UCSF 操作说明](docs/UCSF_EXTERNAL_BASELINE_FULL.md)。这项比较经过**数据集内重新训练**，不是零样本外部测试。真实预测的典型病例图可由 `scripts/select_boundary_typical_cases.py` 和 `scripts/run_multiseed_typical_case_analysis.py` 生成。

## 核查

安装 `pytest` 后运行 `python -m pytest tests -q`。模型权重、原始 MRI、缓存、论文草稿和本机结果均不属于源代码仓库；`.gitignore` 会排除这些文件。不同 AutoDL 环境的 GPU、CUDA 和数据目录仍需按实际配置。
