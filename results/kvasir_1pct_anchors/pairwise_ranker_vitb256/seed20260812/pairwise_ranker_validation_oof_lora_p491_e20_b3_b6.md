# Pairwise route ranker validation OOF (lora_p491_e20, b3-b6)

This is a validation-only, target-grouped nested-CV audit. The test split was not loaded.

| Method | Dice | Delta vs b6 | 95% bootstrap CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_b6 | 0.844065 | +0.000000 | [+0.000000, +0.000000] | +0.000000 | 0/100/0 | 0 | 0 |
| q_multi_only | 0.843688 | -0.000377 | [-0.018364, +0.017212] | +0.001346 | 61/6/33 | 5 | 100 |
| q_model_mean_only | 0.847980 | +0.003915 | [-0.029367, +0.034200] | +0.008175 | 55/2/43 | 10 | 99 |
| b7_mean | 0.852645 | +0.008580 | [-0.017936, +0.032683] | +0.004167 | 60/2/38 | 6 | 100 |
| linear3_outer_train | 0.854734 | +0.010669 | [-0.016962, +0.037143] | +0.005151 | 54/1/45 | 8 | 100 |
| pairwise_quality | 0.877041 | +0.032976 | [+0.007191, +0.060993] | +0.007508 | 66/3/31 | 4 | 100 |
| pairwise_quality_fallback | 0.881702 | +0.037638 | [+0.013015, +0.065104] | +0.007674 | 60/13/27 | 2 | 88 |
| pairwise_qmodel | 0.875222 | +0.031157 | [+0.005691, +0.059616] | +0.011129 | 67/3/30 | 6 | 99 |
| pairwise_qmodel_fallback | 0.875211 | +0.031146 | [+0.005225, +0.059552] | +0.011115 | 65/6/29 | 6 | 95 |

## Concentration diagnostics

| Method | Max gain | Max regression | Largest gain / positive gains | Largest gain / net gain |
|---|---:|---:|---:|---:|
| fixed_b6 | +0.000000 | +0.000000 | 0.000 | n/a |
| q_multi_only | +0.404585 | -0.489266 | 0.298 | n/a |
| q_model_mean_only | +0.558614 | -0.796497 | 0.151 | 1.427 |
| b7_mean | +0.530620 | -0.796497 | 0.189 | 0.618 |
| linear3_outer_train | +0.555525 | -0.764658 | 0.167 | 0.521 |
| pairwise_quality | +0.681855 | -0.384723 | 0.154 | 0.207 |
| pairwise_quality_fallback | +0.681855 | -0.384723 | 0.154 | 0.181 |
| pairwise_qmodel | +0.681855 | -0.384723 | 0.151 | 0.219 |
| pairwise_qmodel_fallback | +0.681855 | -0.384723 | 0.151 | 0.219 |

## Fold choices

| Fold | Linear weights (return/multi/model) | Quality ranker (L2/fallback) | Student ranker (L2/fallback) |
|---:|---:|---:|---:|
| 0 | 0.80/0.00/0.20 | 0.001/0.70 | 0.01/0.75 |
| 1 | 0.45/0.15/0.40 | 0.001/0.50 | 0.1/0.60 |
| 2 | 1.00/0.00/0.00 | 0.01/0.70 | 0.001/0.50 |
| 3 | 1.00/0.00/0.00 | 0.1/0.50 | 0.1/0.50 |
| 4 | 1.00/0.00/0.00 | 0.01/0.75 | 0.001/0.60 |
