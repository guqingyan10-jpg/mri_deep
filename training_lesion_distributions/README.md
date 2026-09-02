# Training-fitted lesion strata

本目录存放在 AutoDL 数据环境中由
[`scripts/derive_train_lesion_strata.py`](../scripts/derive_train_lesion_strata.py)
生成的训练集病灶体素分布与固定分层阈值：

```text
wt_training_lesion_size_distribution.csv
wt_training_lesion_strata.json
et_training_lesion_size_distribution.csv
et_training_lesion_strata.json
```

这些文件只使用 `tumourCSV.csv` 中确定划分后的257例训练病例，默认采用
26邻域并过滤小于10体素的连通域。验证集和测试集不得重新拟合阈值。

生成命令：

```bash
python scripts/derive_train_lesion_strata.py --region BOTH
```

JSON 是正式评价的审计依据；生成后应随对应实验结果保存，不要手工修改。
