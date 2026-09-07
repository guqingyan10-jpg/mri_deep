# Key Model Comparison — Core Indicators

| Model | Macro Dice | ET Dice | ET HD95↓ | Small-case Dice | Lesion F1 |
|---|---|---|---|---|---|
| Baseline (BCEDice) | 0.821 | 0.758 | 10.26 | 0.621 | 0.617 |
| Edge (Laplacian, concat) | 0.817 | 0.781 | 7.36 | 0.672 | 0.568 |
| HF Concat Boundary (Laplacian, w=0.1) | 0.818 | 0.777 | 7.31 | 0.676 | 0.631 |
| HF Concat Boundary + Multi-scale V2 (w=0.1, seed55) | 0.825 | 0.776 | 8.29 | 0.670 | 0.630 |
| HF Concat Boundary (Laplacian, w=0.05) | 0.828 | 0.772 | 6.89 | 0.659 | 0.599 |

## Delta vs Baseline

| Model | Δ Macro Dice | Δ ET Dice | Δ ET HD95 | Δ Small-case Dice | Δ Lesion F1 |
|---|---|---|---|---|---|
| Baseline (BCEDice) | (baseline) | — | — | — | — |
| Edge (Laplacian, concat) | -0.0037 | +0.0229 | -2.90 | +0.0504 | -0.0490 |
| HF Concat Boundary (Laplacian, w=0.1) | -0.0029 | +0.0180 | -2.95 | +0.0551 | +0.0146 |
| HF Concat Boundary + Multi-scale V2 (w=0.1, seed55) | +0.0046 | +0.0179 | -1.97 | +0.0486 | +0.0134 |
| HF Concat Boundary (Laplacian, w=0.05) | +0.0076 | +0.0139 | -3.37 | +0.0374 | -0.0176 |