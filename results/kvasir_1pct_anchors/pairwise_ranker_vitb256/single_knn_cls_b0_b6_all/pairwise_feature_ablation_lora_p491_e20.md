# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

Single fixed family anchor_conditioned_target_pooling__knn_cls at b0-b6 (7 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| route_prior | 0.844065 | +0.000000 | [+0.000000, +0.000000] | +0.000000 | 0/100/0 | 0 | 0 |
| internal_quality | 0.847394 | +0.003329 | [-0.007114, +0.014232] | +0.000065 | 27/40/33 | 3 | 61 |
| internal_plus_consensus | 0.839977 | -0.004088 | [-0.011696, +0.001332] | +0.000052 | 24/50/26 | 3 | 51 |
| quality_plus_relative | 0.851608 | +0.007543 | [-0.005523, +0.028596] | -0.000077 | 32/32/36 | 2 | 68 |
| route_prior_fallback | 0.844065 | +0.000000 | [+0.000000, +0.000000] | +0.000000 | 0/100/0 | 0 | 0 |
| internal_quality_fallback | 0.845354 | +0.001289 | [-0.003703, +0.006500] | +0.000116 | 18/63/19 | 2 | 38 |
| internal_plus_consensus_fallback | 0.841634 | -0.002431 | [-0.008706, +0.001091] | +0.000080 | 17/69/14 | 1 | 32 |
| quality_plus_relative_fallback | 0.853411 | +0.009346 | [-0.001569, +0.029228] | -0.000096 | 24/44/32 | 1 | 56 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
