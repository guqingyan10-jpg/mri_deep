# V2 Alpha Sensitivity (Test Set)

Seed 55 is the main experiment; seeds 42 and 123 use the stability runner. Cross-seed summaries are descriptive because the training protocols differ.

## Learned alpha from evaluated best checkpoints

| Seed | Protocol | Best epoch | Learned alpha |
|---:|---|---:|---:|
| 42 | stability_runner | 51 | 0.05985187 |
| 55 | main_experiment | 90 | -0.036205307 |
| 123 | stability_runner | 33 | -0.030405693 |

Best-checkpoint alpha mean: **-0.0022530432**; range: **[-0.036205307, 0.05985187]**.

## Per-seed sensitivity

| Seed | Protocol | Alpha mode | Eval alpha | Macro Dice | ET Dice | Small-lesion GT-anchored Dice |
|---:|---|---|---:|---:|---:|---:|
| 42 | stability_runner | zero | 0 | 0.82514 | 0.78276 | 0.07837 |
| 42 | stability_runner | learned | 0.05985187 | 0.82509 | 0.78261 | 0.07861 |
| 42 | stability_runner | one | 1 | 0.82058 | 0.77512 | 0.05884 |
| 55 | main_experiment | zero | 0 | 0.82519 | 0.77616 | 0.02349 |
| 55 | main_experiment | learned | -0.036205307 | 0.82520 | 0.77639 | 0.02353 |
| 55 | main_experiment | one | 1 | 0.82280 | 0.77555 | 0.02087 |
| 123 | stability_runner | zero | 0 | 0.81222 | 0.77786 | 0.09704 |
| 123 | stability_runner | learned | -0.030405693 | 0.81222 | 0.77787 | 0.09710 |
| 123 | stability_runner | one | 1 | 0.81069 | 0.77747 | 0.11588 |

## Small-lesion ET diagnostics

| Seed | Alpha mode | GT lesions | Detected | Missed | Recall | Miss rate | Matched Dice | GT-anchored Dice |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | zero | 31 | 5 | 26 | 0.16129 | 0.83871 | 0.48592 | 0.07837 |
| 42 | learned | 31 | 5 | 26 | 0.16129 | 0.83871 | 0.48736 | 0.07861 |
| 42 | one | 31 | 4 | 27 | 0.12903 | 0.87097 | 0.45598 | 0.05884 |
| 55 | zero | 31 | 2 | 29 | 0.06452 | 0.93548 | 0.36406 | 0.02349 |
| 55 | learned | 31 | 2 | 29 | 0.06452 | 0.93548 | 0.36471 | 0.02353 |
| 55 | one | 31 | 2 | 29 | 0.06452 | 0.93548 | 0.32355 | 0.02087 |
| 123 | zero | 31 | 10 | 21 | 0.32258 | 0.67742 | 0.30083 | 0.09704 |
| 123 | learned | 31 | 10 | 21 | 0.32258 | 0.67742 | 0.30100 | 0.09710 |
| 123 | one | 31 | 11 | 20 | 0.35484 | 0.64516 | 0.32656 | 0.11588 |

## Descriptive cross-seed summary

| Alpha mode | Macro Dice | ET Dice | Small-lesion GT-anchored Dice |
|---|---:|---:|---:|
| zero | 0.82085 +/- 0.00748 | 0.77893 +/- 0.00342 | 0.06630 +/- 0.03823 |
| learned | 0.82084 +/- 0.00746 | 0.77896 +/- 0.00325 | 0.06641 +/- 0.03827 |
| one | 0.81802 +/- 0.00645 | 0.77605 +/- 0.00125 | 0.06520 +/- 0.04782 |
