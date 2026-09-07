# ResUNet Enhancement — Experimental Results

## Core Metrics (primary indicators)

| Model | Macro Dice | ET Dice | ET HD95↓ | Small-case ET Dice |
|---|---|---|---|---|
| HF Concat Boundary + Multi-scale V2 (w=0.1, seed55) | 0.825 | 0.776 | 8.3 | 0.670 |
## Table 1: Complete Evaluation Metrics

| Model | ET Dice | ET Recall | ET Prec. | ET HD95↓ | ET NSD↑ | TC Dice | TC HD95↓ | WT Dice | Lesion Rec. | Lesion Prec. | Lesion F1 | Small ET Dice | #Params | Infer.(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HF Concat Boundary + Multi-scale V2 (w=0.1, seed55) | 0.776 | 0.794 | 0.812 | 8.3 | 0.787 | 0.805 | 8.9 | 0.894 | 0.661 | 0.728 | 0.630 | 0.670 | 7,102,735 | 0.14 |


## Table 2: Delta vs Baseline

| Model | Δ ET Dice | Δ ET Recall | Δ ET Prec. | Δ ET HD95 | Δ ET NSD | Δ Lesion Rec. | Δ Small ET Dice |
|---|---|---|---|---|---|---|---|
| HF Concat Boundary + Multi-scale V2 (w=0.1, seed55) | (baseline) | — | — | — | — | — | — |


## Table 3: Best Model by Category

| Category | Best Model | ET Dice | ET HD95 | Small ET Dice |
|---|---|---|---|---|
| Final Combination | HF Concat Boundary + Multi-scale V2 (w=0.1, seed55) | 0.776 | 8.3 | 0.670 |