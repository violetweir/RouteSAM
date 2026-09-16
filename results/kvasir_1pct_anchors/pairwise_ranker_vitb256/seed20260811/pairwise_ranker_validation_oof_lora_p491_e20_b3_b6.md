# Pairwise route ranker validation OOF (lora_p491_e20, b3-b6)

This is a validation-only, target-grouped nested-CV audit. The test split was not loaded.

| Method | Dice | Delta vs b6 | 95% bootstrap CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_b6 | 0.844065 | +0.000000 | [+0.000000, +0.000000] | +0.000000 | 0/100/0 | 0 | 0 |
| q_multi_only | 0.843688 | -0.000377 | [-0.018125, +0.016122] | +0.001346 | 61/6/33 | 5 | 100 |
| q_model_mean_only | 0.847980 | +0.003915 | [-0.028691, +0.034794] | +0.008175 | 55/2/43 | 10 | 99 |
| b7_mean | 0.852645 | +0.008580 | [-0.017360, +0.033022] | +0.004167 | 60/2/38 | 6 | 100 |
| linear3_outer_train | 0.857150 | +0.013085 | [-0.015237, +0.040569] | +0.006761 | 53/1/46 | 8 | 100 |
| pairwise_quality | 0.885300 | +0.041235 | [+0.018957, +0.068132] | +0.006187 | 66/2/32 | 0 | 100 |
| pairwise_quality_fallback | 0.885080 | +0.041015 | [+0.018479, +0.067177] | +0.005912 | 59/11/30 | 0 | 90 |
| pairwise_qmodel | 0.880536 | +0.036471 | [+0.012243, +0.063729] | +0.009098 | 59/4/37 | 3 | 98 |
| pairwise_qmodel_fallback | 0.879422 | +0.035357 | [+0.011119, +0.063038] | +0.007153 | 47/27/26 | 3 | 73 |

## Concentration diagnostics

| Method | Max gain | Max regression | Largest gain / positive gains | Largest gain / net gain |
|---|---:|---:|---:|---:|
| fixed_b6 | +0.000000 | +0.000000 | 0.000 | n/a |
| q_multi_only | +0.404585 | -0.489266 | 0.298 | n/a |
| q_model_mean_only | +0.558614 | -0.796497 | 0.151 | 1.427 |
| b7_mean | +0.530620 | -0.796497 | 0.189 | 0.618 |
| linear3_outer_train | +0.555525 | -0.764658 | 0.151 | 0.425 |
| pairwise_quality | +0.681855 | -0.041146 | 0.158 | 0.165 |
| pairwise_quality_fallback | +0.681855 | -0.041146 | 0.159 | 0.166 |
| pairwise_qmodel | +0.681855 | -0.291249 | 0.151 | 0.187 |
| pairwise_qmodel_fallback | +0.681855 | -0.291249 | 0.158 | 0.193 |

## Fold choices

| Fold | Linear weights (return/multi/model) | Quality ranker (L2/fallback) | Student ranker (L2/fallback) |
|---:|---:|---:|---:|
| 0 | 1.00/0.00/0.00 | 0.01/0.65 | 0.1/0.70 |
| 1 | 0.80/0.00/0.20 | 0.1/0.50 | 0.001/0.70 |
| 2 | 1.00/0.00/0.00 | 0.1/0.55 | 1/0.55 |
| 3 | 0.60/0.10/0.30 | 0.1/0.70 | 0.1/0.60 |
| 4 | 1.00/0.00/0.00 | 1/0.50 | 0.1/0.75 |
