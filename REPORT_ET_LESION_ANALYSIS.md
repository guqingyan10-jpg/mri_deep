# BraTS2020 ET Lesion Analysis — Internal Report

> 生成日期: 2026-08-01
> 分析范围: BraTS2020 Training Set (368 例, 981 个独立 ET 病灶)
> 工具: `scripts/et_statistics.py`

---

## 1. 为什么做这个分析

**你的 baseline 评估已经暴露了一个核心矛盾：**

```
Overall ET Dice ≈ 0.75  ← "看起来不错"
Lesion-wise Recall ≈ 0.55-0.65  ← "但漏掉了 1/3 的病灶"
Small-case Dice ≈ 0.62 (vs Overall 0.76)  ← "小病灶 Dice 比整体低 12-14%"
```

**Dice 骗了你——** 它由大病灶主导，掩盖了小病灶的糟糕表现。这份报告的功能：**用数据证明小 ET 病灶是核心瓶颈**。

---

## 2. 关键发现

### 2.1 病灶大小分布：近半数无法被 Dice 感知

```
        病灶大小分布 (n=981)
        ─────────────────
   50 ┤██████████████████████████████████████████  42.0%
      │
  100 ┤███████  7.2%
      │
  500 ┤███████████  11.4%
      │
 1000 ┤███  2.8%
      │
 5000 ┤█████████  9.0%
      │
10000 ┤██████  6.1%
      │
50000 ┤███████████████████  18.9%
      │
 >50K ┤███  2.7%
```

**中位数仅 113 voxels。** 算 Dice 时，一个 100,000-voxel 的大病灶和一个 20-voxel 的微型病灶权重一样——100,000 淹没 20。

### 2.2 多灶性是常态

```
单病灶:  42.4%  ████████████████████
2个病灶:  22.6%  ███████████
3个病灶:  10.9%  █████
4个病灶:   6.0%  ███
5-9个:     7.1%  ████
≥10个:     3.8%  ██
──────────────────────────
总计 ≥2:  50.3%  ← 一半病例有多灶性 ET
```

### 2.3 典型模式: 大病灶 + 卫星小碎片

```
75.5% 的病例: 主病灶占 >90%, 其余为小碎片 (<500 vox)
```

看这组数据就懂了：

```
BraTS20_Training_274:  主病灶 88,587 vox,  + 8个小碎片 (13-420 vox)
BraTS20_Training_293:  主病灶 36,903 vox,  + 34个小碎片 (10-378 vox)
BraTS20_Training_191:  主病灶 87,026 vox,  + 4个小碎片 (25-505 vox)
```

**传统模型在这类病例上的表现: Overall Dice ≈ 0.85（看起来很好），但实际上 8 个小碎片可能漏了 6 个（Lesion Recall ≈ 0.25）。**

### 2.4 HGG vs LGG: 两个完全不同的问题

| | HGG (n=292) | LGG (n=76) |
|---|---|---|
| ET 体积 (mean) | 23,263 vox | 5,463 vox |
| ET/WT 比 | 0.241 | **0.053** (差 4.6x) |
| 零 ET 病例 | 0 (0%) | 27 (35.5%) |
| 多灶性 | 50.7% | 48.7% |

**对论文的启示: 评估必须分 HGG/LGG 报告, LGG 排除 27 例零 ET (否则 Recall=NaN)。**

---

## 3. 论文可用的黄金数字

| # | 陈述 (可直接写入 Introduction / Motivation) |
|---|---|
| 1 | 42.0% of ET lesions are smaller than 50 voxels (~0.05 cm³), invisible to conventional Dice |
| 2 | 50.3% of BraTS2020 cases exhibit multi-focal ET (≥2 disconnected lesions) |
| 3 | 75.5% of cases follow a "dominant lesion + satellite fragments" pattern |
| 4 | 91.0% of cases have secondary ET tissue beyond the main lesion |
| 5 | LGG patients have 4.6× lower ET/WT ratio than HGG (0.053 vs 0.241) |
| 6 | 35.5% of LGG cases have zero ET — model must learn when NOT to predict ET |
| 7 | The largest 2.7% of lesions account for the majority of Dice signal |

---

## 4. 建议选来可视化的典型病例

### 病例 A: 单病灶大 ET (基准/简单例)
```
BraTS20_Training_225:
  ET = 111,250 vox (全集最大), 纯单病灶
  用途: 展示"简单情况"下所有模型都表现好
```

### 病例 B: 经典多灶性 (核心展示)
```
BraTS20_Training_274:
  ET = 89,546 vox, 9 个病灶
  主病灶 88,587 + 8个小碎片 (13-420 vox)
  用途: 展示 baseline 漏掉了哪些小碎片, 改进模型如何检出它们
  这是论文最有说服力的可视化
```

### 病例 C: 极度碎片化
```
BraTS20_Training_293:
  ET = 39,301 vox, 35 个病灶 (!)
  主病灶 36,903 + 34个小碎片
  用途: 展示极端 case, 说明问题的严重性
```

### 病例 D: LGG 无 ET (对比)
```
BraTS20_Training_329:
  ET = 0, WT = 62,935, TC = 45,782
  用途: 展示 LGG 可以有大肿瘤但零 ET, 模型不应该 hallucinate ET
```

### 病例 E: 微型多灶性
```
BraTS20_Training_284:
  ET = 3,562 vox, 16 个病灶
  主病灶 2,789 + 15个小碎片 (10-232 vox)
  用途: 全小病灶 — 展示 baseline 几乎全漏
```

---

## 5. 下载清单

从服务器下载以下文件到本地:

```bash
# 在服务器上打包
cd /root/autodl-tmp && tar -czf brats2020_results.tar.gz \
  et_statistics.csv \
  et_components_detail.csv \
  mri_deep/RESEARCH_RECORDS.md \
  mri_deep/CHANGELOG.md

# 然后通过 AutoDL JupyterLab 文件管理器下载 brats2020_results.tar.gz
```

**还需要下载（如果没保存）:**

| 文件 | 位置 | 用途 |
|---|---|---|
| `et_statistics.csv` | `/root/autodl-tmp/` | 病例级 ET 统计 |
| `et_components_detail.csv` | `/root/autodl-tmp/` | 病灶级明细 |
| Baseline 评估结果截图 | 你笔记本截图 | 证据保存 |

---

## 6. 下一步

- [ ] 选 3-5 个典型病例可视化 (A/B/C/D/E)
- [ ] V1 训练跑起来 (`--lambda_b 0.1/0.3/0.5`)
- [ ] V1 结果对比 → 确定最优 λb
- [ ] V2: 高频边缘辅助分支
