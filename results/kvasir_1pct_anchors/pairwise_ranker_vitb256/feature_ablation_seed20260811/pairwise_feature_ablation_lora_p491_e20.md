# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

P2 contains all 11 route families at b3-b6 (44 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| internal_plus_consensus | 0.892173 | +0.048108 | [+0.024858, +0.075778] | +0.012548 | 62/3/35 | 0 | 100 |
| internal_plus_consensus_fallback | 0.888328 | +0.044263 | [+0.021168, +0.070982] | +0.007945 | 50/21/29 | 0 | 81 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
