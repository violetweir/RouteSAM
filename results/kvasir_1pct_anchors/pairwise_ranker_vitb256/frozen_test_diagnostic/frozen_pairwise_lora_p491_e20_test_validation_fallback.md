# Frozen pairwise ranker test diagnostic (lora_p491_e20)

The frozen validation-trained model was applied without fitting, calibration, or student predictions.

- selected Dice: **0.906634**
- fixed b6 Dice: **0.897780**
- delta: **+0.008854**; bootstrap 95% CI [-0.014215, +0.031496]
- oracle: 0.934728; gap 0.028094
- wins/ties/losses: 50/12/38
- catastrophic regressions: 4
- trimmed mean delta (10%): +0.001169
- largest gain/net gain: 0.7129426573927529
- validation-derived fallback: True at probability 0.65

For Kvasir test this remains a historical diagnostic because earlier selector experiments already inspected the split.
