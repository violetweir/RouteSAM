# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

Single fixed family anchor_conditioned_target_pooling__knn_cls at b1-b6 (6 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| internal_plus_consensus | 0.857833 | +0.013768 | [-0.001432, +0.037623] | +0.000327 | 34/33/33 | 3 | 67 |
| internal_plus_consensus_fallback | 0.856593 | +0.012528 | [-0.002433, +0.035616] | +0.000188 | 29/42/29 | 3 | 58 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
