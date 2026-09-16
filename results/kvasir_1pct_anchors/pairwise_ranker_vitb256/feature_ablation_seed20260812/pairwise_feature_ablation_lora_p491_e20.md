# Pairwise ranker feature ablation (lora_p491_e20, validation OOF)

P2 contains all 11 route families at b3-b6 (44 candidates per target). Test was not loaded.

| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| internal_plus_consensus | 0.891281 | +0.047216 | [+0.023398, +0.074452] | +0.013226 | 59/5/36 | 2 | 100 |
| internal_plus_consensus_fallback | 0.891281 | +0.047216 | [+0.023860, +0.074730] | +0.013226 | 59/5/36 | 2 | 99 |

Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement.
