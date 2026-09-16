# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

Single fixed family anchor_conditioned_target_pooling__knn_cls at b1-b6 (6 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| route_prior | 0.844065 | +0.000000 | [+0.000000, +0.000000] | +0.000000 | 0/100/0 | 0 | 0 |
| internal_quality | 0.849416 | +0.005351 | [-0.001162, +0.014105] | +0.000637 | 29/45/26 | 2 | 55 |
| internal_plus_consensus | 0.860942 | +0.016877 | [+0.001761, +0.039717] | +0.000893 | 31/41/28 | 1 | 59 |
| quality_plus_relative | 0.857367 | +0.013302 | [-0.002896, +0.036693] | +0.001067 | 42/25/33 | 3 | 75 |
| route_prior_fallback | 0.844065 | +0.000000 | [+0.000000, +0.000000] | +0.000000 | 0/100/0 | 0 | 0 |
| internal_quality_fallback | 0.849448 | +0.005383 | [-0.001230, +0.013726] | +0.000499 | 24/53/23 | 2 | 47 |
| internal_plus_consensus_fallback | 0.857192 | +0.013127 | [-0.000039, +0.035950] | +0.000156 | 19/62/19 | 1 | 38 |
| quality_plus_relative_fallback | 0.857513 | +0.013449 | [-0.002882, +0.036812] | +0.000981 | 35/40/25 | 3 | 60 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
