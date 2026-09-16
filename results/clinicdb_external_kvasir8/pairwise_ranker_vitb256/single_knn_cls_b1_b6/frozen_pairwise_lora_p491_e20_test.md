# Frozen pairwise ranker test diagnostic (lora_p491_e20)

The frozen validation-trained model was applied without fitting, calibration, or student predictions.

- selected Dice: **0.854612**
- fixed b6 Dice: **0.815820**
- delta: **+0.038792**; bootstrap 95% CI [+0.000295, +0.087529]
- oracle: 0.866798; gap 0.012187
- wins/ties/losses: 26/18/17
- catastrophic regressions: 3
- trimmed mean delta (10%): +0.001830
- largest gain/net gain: 0.402095100484385
- validation-derived fallback: False

For Kvasir test this remains a historical diagnostic because earlier selector experiments already inspected the split.
