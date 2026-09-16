# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

P2 contains all 11 route families at b3-b6 (44 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| route_prior | 0.854522 | +0.010457 | [-0.012094, +0.034434] | +0.002709 | 54/0/46 | 7 | 100 |
| internal_quality | 0.879969 | +0.035904 | [+0.013941, +0.060389] | +0.010997 | 59/0/41 | 3 | 100 |
| internal_plus_consensus | 0.891082 | +0.047017 | [+0.023583, +0.074093] | +0.012544 | 63/1/36 | 1 | 100 |
| quality_plus_relative | 0.880443 | +0.036378 | [+0.011928, +0.064072] | +0.008407 | 61/4/35 | 3 | 100 |
| quality_plus_qmodel | 0.874183 | +0.030118 | [+0.004583, +0.057567] | +0.005769 | 62/2/36 | 5 | 100 |
| route_prior_fallback | 0.855607 | +0.011542 | [-0.005879, +0.031040] | +0.001885 | 46/20/34 | 5 | 80 |
| internal_quality_fallback | 0.879735 | +0.035670 | [+0.013635, +0.060123] | +0.010691 | 54/6/40 | 3 | 94 |
| internal_plus_consensus_fallback | 0.890588 | +0.046523 | [+0.023237, +0.073492] | +0.011904 | 58/8/34 | 1 | 93 |
| quality_plus_relative_fallback | 0.879641 | +0.035576 | [+0.010230, +0.062972] | +0.007404 | 48/23/29 | 3 | 79 |
| quality_plus_qmodel_fallback | 0.874543 | +0.030478 | [+0.005124, +0.059083] | +0.005583 | 49/26/25 | 5 | 74 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
