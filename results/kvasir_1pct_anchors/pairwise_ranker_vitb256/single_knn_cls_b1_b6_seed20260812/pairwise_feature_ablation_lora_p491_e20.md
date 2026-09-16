# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

Single fixed family anchor_conditioned_target_pooling__knn_cls at b1-b6 (6 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| internal_plus_consensus | 0.860045 | +0.015980 | [+0.000758, +0.039090] | +0.000692 | 37/36/27 | 1 | 64 |
| internal_plus_consensus_fallback | 0.850611 | +0.006546 | [+0.000468, +0.014762] | +0.000237 | 22/58/20 | 0 | 42 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
