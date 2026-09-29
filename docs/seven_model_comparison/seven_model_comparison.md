| 方法 | Macro Dice ↑ | ET Dice ↑ | ET HD95 (mm) ↓ | Boundary Dice (ET NSD, 1 mm) ↑ | 病灶 F1 ↑ | 小病灶 Dice (GT 锚定) ↑ | 小病灶 Dice (matched) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3D U-Net | 0.7935 | 0.7494 | 16.24 | 0.7415 | 0.4315 | 0.0687 | — |
| 3D ResUNet | 0.8207 | 0.7585 | 10.26 | 0.7503 | 0.5213 | 0.0357 | 0.3685 |
| 3D Attention U-Net | 0.7743 | 0.7317 | 16.93 | 0.7062 | 0.5081 | 0.0076 | — |
| nnU-Net 风格 3D 网络 | 0.7972 | 0.7577 | 11.11 | 0.7640 | 0.5143 | 0.0336 | — |
| AFBMS-ResUNet | 0.8252 | 0.7764 | 8.29 | 0.7871 | 0.5385 | 0.0235 | 0.3647 |
| DoubleBlock-ViT | — | — | — | — | — | — | — |
| SuperLightNet | — | — | — | — | — | — | — |

口径：BraTS2020 固定 37 例测试集，阈值 0.33。Boundary Dice 沿用旧表的 ET normalized surface Dice（NSD，容差 1 mm）；病灶 F1 为 ET 26 连通、最小 10 体素的一对一匹配后 pooled/micro F1。小病灶为训练集预先确定的 10–44 体素 GT ET 病灶；GT 锚定 Dice 对漏检计 0，matched Dice 仅平均成功匹配的 GT 病灶。— 表示尚未取得可验证结果。

前五行原有六列逐字保留用户给出的四位小数，不代表重新评估；两个新增模型须以各自训练完成的 best_model_*.pth 计算。
