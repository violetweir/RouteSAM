# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

Single fixed family anchor_conditioned_target_pooling__knn_cls at b1-b6 (6 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| internal_plus_consensus | 0.860942 | +0.016877 | [+0.001976, +0.040033] | +0.000893 | 31/41/28 | 1 | 59 |
| internal_plus_consensus_fallback | 0.857192 | +0.013127 | [-0.000142, +0.035447] | +0.000156 | 19/62/19 | 1 | 38 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
