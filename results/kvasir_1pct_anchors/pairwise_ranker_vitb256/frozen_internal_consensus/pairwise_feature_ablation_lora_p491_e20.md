# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

P2 contains all 11 route families at b3-b6 (44 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| internal_plus_consensus | 0.891082 | +0.047017 | [+0.023660, +0.074432] | +0.012544 | 63/1/36 | 1 | 100 |
| internal_plus_consensus_fallback | 0.890588 | +0.046523 | [+0.023219, +0.073946] | +0.011904 | 58/8/34 | 1 | 93 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
