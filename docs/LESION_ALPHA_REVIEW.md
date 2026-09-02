# WT/ET 病灶分层与 α 敏感性：审核指南

这份文档是近期新增评价代码的审核入口。正式结果默认评价固定的37例
测试集；`valid_test` 仅用于扩大样本量的探索性分析，不用于替代正式测试
结果。模型权重一律使用根据验证损失保存的 `best_model_*.pth`。

## 1. 代码导航

| 文件 | 职责 |
|---|---|
| [`evaluation/wt_lesion_stratified.py`](../evaluation/wt_lesion_stratified.py) | WT/ET 共用的分层、26邻域连通域、一对一病灶匹配与汇总指标。文件名保留历史 WT 命名，但实现是区域无关的。 |
| [`scripts/derive_train_lesion_strata.py`](../scripts/derive_train_lesion_strata.py) | 仅从实际训练集提取 WT/ET GT 病灶体素数，拟合并保存固定 small/medium/large 阈值。 |
| [`scripts/eval_wt_lesion_stratified.py`](../scripts/eval_wt_lesion_stratified.py) | 五个模型的 WT 病灶级正式评价；默认 `test`，也提供 `valid` 与探索性 `valid_test`。 |
| [`scripts/eval_et_lesion_stratified.py`](../scripts/eval_et_lesion_stratified.py) | ET 入口，复用上一文件的完全相同评价流程。 |
| [`scripts/eval_alpha_sensitivity.py`](../scripts/eval_alpha_sensitivity.py) | 三个种子 V2 best checkpoint 的 α=0 / learned / 1 敏感性；固定37例测试集。 |
| [`tests/test_wt_lesion_stratified.py`](../tests/test_wt_lesion_stratified.py) | 阈值边界、一对一匹配、漏检、空GT病例与病灶级池化测试。 |
| [`tests/test_wt_eval_model_registry.py`](../tests/test_wt_eval_model_registry.py) | WT/ET入口、数据阶段、模型注册和 best-checkpoint 约束测试。 |
| [`tests/test_alpha_sensitivity.py`](../tests/test_alpha_sensitivity.py) | α模式、测试集、输出审计和small病灶指标测试。 |

## 2. 固定数据协议

`data/dataset.py` 从 `tumourCSV.csv` 以 `random_state=10` 做确定性划分：

- train：257例，且只用这部分拟合病灶大小阈值；
- valid：74例，不参与阈值拟合；
- test：37例，正式模型评价与当前 α 敏感性使用；
- valid + test：111例，只作为探索性合并分析，保留每例原始 split 标签。

任何验证集或测试集的预测、Dice和病灶检出结果都不会参与阈值拟合。
阈值 JSON 必须声明 `fit_split=train`、26-connectivity 和与运行参数一致的
`min_component_size=10`，否则评价脚本拒绝运行。

## 3. 病灶分层与匹配定义

1. 对裁剪后的三维 GT 掩膜使用26邻域连通域，过滤小于10体素的组件。
2. WT 使用标签 `{1,2,4}`；ET 使用标签 `{4}`。
3. 在训练集病灶体素数的所有可行整数切点中选择 small/medium/large 数量
   尽量接近的切点；相同体素数的病灶不会被拆到不同档。
4. 阈值拟合完成后冻结，验证集、测试集与 `valid_test` 都使用同一组阈值。
5. 分档依据始终是 GT 病灶大小，而不是预测病灶大小。
6. GT 与预测连通域按 Dice 最大化进行 Hungarian 一对一匹配，并且要求
   实际交集大于0；一个预测不能同时检出两个 GT 病灶。

每一档输出的核心病灶级指标：

- `lesion_recall = detected GT lesions / all GT lesions`；
- `miss_rate = missed GT lesions / all GT lesions`；
- `matched_lesion_dice`：只对成功匹配的病灶求平均；
- `gt_anchored_lesion_dice`：每个漏检 GT 病灶记 Dice=0 后，对所有 GT
  病灶求平均；
- `gt_lesions`、`detected`、`missed`：该档实际病灶数量。

因此这些指标按病灶池化，不是先按病例求指标再平均。没有 GT 病灶的病例
不会被记为“召回率1”，而是单独记录为空GT、仅假阳性或真阴性病例。

## 4. WT/ET 正式评价复现

先在 AutoDL 数据环境中生成训练集固定阈值：

```bash
python scripts/derive_train_lesion_strata.py \
  --csv tumourCSV.csv \
  --region BOTH \
  --min-component-size 10
```

再使用同一阈值评价37例测试集：

```bash
python scripts/eval_wt_lesion_stratified.py \
  --phase test \
  --strata-json training_lesion_distributions/wt_training_lesion_strata.json

python scripts/eval_et_lesion_stratified.py \
  --phase test \
  --strata-json training_lesion_distributions/et_training_lesion_strata.json
```

探索性 `valid + test` 合并版必须显式请求：

```bash
python scripts/eval_wt_lesion_stratified.py \
  --phase valid_test \
  --strata-json training_lesion_distributions/wt_training_lesion_strata.json

python scripts/eval_et_lesion_stratified.py \
  --phase valid_test \
  --strata-json training_lesion_distributions/et_training_lesion_strata.json
```

正式 `test` 与探索性 `valid_test` 使用不同默认输出目录，且均输出病例清单、
逐病灶明细、汇总 CSV/JSON、训练阈值副本和对比图。

## 5. α 敏感性复现

每个种子固定同一个 V2 best checkpoint，三次推理前均重新加载该 state，
只覆盖标量 `multiscale_context.alpha`：0、checkpoint learned α、1。脚本
严格要求 `best_model_*.pth`，不会回退到 last epoch，并强制测试集为37例。

默认 checkpoint 映射：

| Seed | 目录 | 协议标签 |
|---:|---|---|
| 42 | `/root/autodl-tmp/stability/seed42/hf_concat_boundary_w0.1_multiscale_v2` | stability runner |
| 55 | `/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model` | main experiment |
| 123 | `/root/autodl-tmp/stability/seed123/hf_concat_boundary_w0.1_multiscale_v2` | stability runner |

先审计 checkpoint，再运行：

```bash
python scripts/eval_alpha_sensitivity.py --inspect-only

python scripts/eval_alpha_sensitivity.py \
  --et-strata-json training_lesion_distributions/et_training_lesion_strata.json
```

三项主指标中，Macro Dice 和 ET Dice 保持项目原有的病例级定义；
`Small-lesion ET GT-anchored Dice` 是病灶级指标，并复用第3节的一对一匹配
和训练集固定 ET 阈值。脚本还输出small病灶的 Matched Dice、Recall、
Miss rate及逐病灶明细。

默认输出目录为 `alpha_sensitivity_test_results/`。重点审计文件：

- `alpha_checkpoint_values.csv`：三个实际 checkpoint、best epoch与learned α；
- `alpha_sensitivity_per_seed.csv`：三种 α 设置的主指标与相对 learned 差值；
- `alpha_sensitivity_per_lesion.csv`：每个small/medium/large GT病灶的匹配；
- `small_et_test_lesions.csv`：固定测试集small ET病灶清单；
- `et_lesion_strata_applied.json`：所用训练阈值及 `apply_split=test`；
- `alpha_sensitivity_report.md` 与 `alpha_sensitivity_metrics.png`：汇总报告和图。

注意：seed55来自主实验，seed42/123来自stability runner，三种子均值与标准差
属于描述性汇总；输出保留协议标签，不将其伪装为完全相同训练协议的重复实验。

## 6. 自动化检查

无需 MRI 数据和 checkpoint 的核心单元测试：

```bash
python -m pytest \
  tests/test_wt_lesion_stratified.py \
  tests/test_wt_eval_model_registry.py \
  tests/test_alpha_sensitivity.py \
  -q
```

审核正式结果时还应确认：

- 控制台和输出 CSV 中的 checkpoint 均为 `best_model_*.pth`；
- test 病例清单正好37例且无重复；
- WT/ET strata JSON 均为训练集拟合并使用相同 `random_state=10`；
- `matched_lesion_dice` 与 `gt_anchored_lesion_dice` 没有混用；
- 正式test结果与探索性 `valid_test` 结果没有放入同一输出目录或表格。
