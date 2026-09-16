# Frozen pairwise ranker test diagnostic (lora_p491_e20)

The frozen validation-trained model was applied without fitting, calibration, or student predictions.

- selected Dice: **0.906307**
- fixed b6 Dice: **0.897780**
- delta: **+0.008527**; bootstrap 95% CI [-0.014469, +0.031172]
- oracle: 0.934728; gap 0.028421
- wins/ties/losses: 53/6/41
- catastrophic regressions: 4
- trimmed mean delta (10%): +0.000965
- largest gain/net gain: 0.7402667145287285
- validation-derived fallback: False

For Kvasir test this remains a historical diagnostic because earlier selector experiments already inspected the split.
