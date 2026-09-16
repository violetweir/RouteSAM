# Frozen pairwise ranker test diagnostic (lora_p491_e20)

The frozen validation-trained model was applied without fitting, calibration, or student predictions.

- selected Dice: **0.857032**
- fixed b6 Dice: **0.815820**
- delta: **+0.041213**; bootstrap 95% CI [-0.005220, +0.096811]
- oracle: 0.912831; gap 0.055798
- wins/ties/losses: 36/3/22
- catastrophic regressions: 2
- trimmed mean delta (10%): +0.002896
- largest gain/net gain: 0.3776922657777908
- validation-derived fallback: False

For Kvasir test this remains a historical diagnostic because earlier selector experiments already inspected the split.
