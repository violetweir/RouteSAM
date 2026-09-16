# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

Single fixed family anchor_conditioned_target_pooling__knn_cls at b0-b6 (7 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| internal_plus_consensus | 0.839977 | -0.004088 | [-0.011728, +0.001325] | +0.000052 | 24/50/26 | 3 | 51 |
| internal_plus_consensus_fallback | 0.841634 | -0.002431 | [-0.008745, +0.001058] | +0.000080 | 17/69/14 | 1 | 32 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
