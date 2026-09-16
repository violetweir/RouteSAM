# Pairwise route ranker validation OOF (lora_p491_e20, b3-b6)

This is a validation-only, target-grouped nested-CV audit. The test split was not loaded.

| Method | Dice | Delta vs b6 | 95% bootstrap CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_b6 | 0.844065 | +0.000000 | [+0.000000, +0.000000] | +0.000000 | 0/100/0 | 0 | 0 |
| q_multi_only | 0.843688 | -0.000377 | [-0.018708, +0.016745] | +0.001346 | 61/6/33 | 5 | 100 |
| q_model_mean_only | 0.847980 | +0.003915 | [-0.029435, +0.034698] | +0.008175 | 55/2/43 | 10 | 99 |
| b7_mean | 0.852645 | +0.008580 | [-0.017811, +0.032308] | +0.004167 | 60/2/38 | 6 | 100 |
| linear3_outer_train | 0.851824 | +0.007759 | [-0.025059, +0.037899] | +0.005155 | 53/2/45 | 6 | 100 |
| pairwise_quality | 0.880443 | +0.036378 | [+0.011465, +0.063539] | +0.008407 | 61/4/35 | 3 | 100 |
| pairwise_quality_fallback | 0.879641 | +0.035576 | [+0.010910, +0.063587] | +0.007404 | 48/23/29 | 3 | 79 |
| pairwise_qmodel | 0.877096 | +0.033031 | [+0.009393, +0.060074] | +0.010763 | 59/4/37 | 5 | 99 |
| pairwise_qmodel_fallback | 0.877835 | +0.033771 | [+0.009295, +0.060649] | +0.011106 | 53/16/31 | 5 | 86 |

## Concentration diagnostics

| Method | Max gain | Max regression | Largest gain / positive gains | Largest gain / net gain |
|---|---:|---:|---:|---:|
| fixed_b6 | +0.000000 | +0.000000 | 0.000 | n/a |
| q_multi_only | +0.404585 | -0.489266 | 0.298 | n/a |
| q_model_mean_only | +0.558614 | -0.796497 | 0.151 | 1.427 |
| b7_mean | +0.530620 | -0.796497 | 0.189 | 0.618 |
| linear3_outer_train | +0.558614 | -0.796497 | 0.160 | 0.720 |
| pairwise_quality | +0.681855 | -0.384723 | 0.152 | 0.187 |
| pairwise_quality_fallback | +0.681855 | -0.384723 | 0.155 | 0.192 |
| pairwise_qmodel | +0.681855 | -0.384723 | 0.155 | 0.206 |
| pairwise_qmodel_fallback | +0.681855 | -0.384723 | 0.155 | 0.202 |

## Fold choices

| Fold | Linear weights (return/multi/model) | Quality ranker (L2/fallback) | Student ranker (L2/fallback) |
|---:|---:|---:|---:|
| 0 | 1.00/0.00/0.00 | 0.1/0.50 | 0.1/0.75 |
| 1 | 0.00/0.25/0.75 | 0.01/0.50 | 0.01/0.50 |
| 2 | 0.80/0.00/0.20 | 1/0.55 | 0.01/0.50 |
| 3 | 1.00/0.00/0.00 | 0.001/0.75 | 0.1/0.65 |
| 4 | 1.00/0.00/0.00 | 0.1/0.75 | 0.001/0.75 |
