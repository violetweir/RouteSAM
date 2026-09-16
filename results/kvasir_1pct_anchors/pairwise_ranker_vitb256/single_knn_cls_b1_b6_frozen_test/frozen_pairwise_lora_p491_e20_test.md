# Frozen pairwise ranker test diagnostic (lora_p491_e20)

The frozen validation-trained model was applied without fitting, calibration, or student predictions.

- selected Dice: **0.884591**
- fixed b6 Dice: **0.897780**
- delta: **-0.013189**; bootstrap 95% CI [-0.037232, +0.006498]
- oracle: 0.913355; gap 0.028764
- wins/ties/losses: 37/28/35
- catastrophic regressions: 6
- trimmed mean delta (10%): -0.000239
- largest gain/net gain: None
- validation-derived fallback: False

For Kvasir test this remains a historical diagnostic because earlier selector experiments already inspected the split.
