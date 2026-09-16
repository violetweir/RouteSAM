# Validation OOF Selector Matrix v2

Corrected: q_multi is recomputed from existing route masks; no SAM3 rerun. 5-fold grouped by query_id.

## Canvas 512
| Feature | Current | Z-score | Geometry | Logistic OOF | Always B3 | Oracle | Logistic-Z Δ 95% CI | Decision hint |
|---|---:|---:|---:|---:|---:|---:|---|---|
| T18 corrected | 0.804241 | 0.806739 | 0.791819 | 0.802060 | 0.804409 | 0.842054 | -0.004678 [-0.012300, +0.000631] | prefer z-score |
| DINO global | 0.801313 | 0.801218 | 0.730830 | 0.803899 | 0.767032 | 0.848924 | +0.002682 [-0.016105, +0.022644] | prefer z-score |
| DINO patch average | 0.777063 | 0.776978 | 0.770175 | 0.799450 | 0.782758 | 0.846125 | +0.022471 [-0.002593, +0.055674] | prefer z-score |
| Target pooling | 0.766048 | 0.765609 | 0.800039 | 0.799513 | 0.770852 | 0.848740 | +0.033904 [+0.001172, +0.072722] | consider logistic |
| Patch correspondence | 0.793123 | 0.790088 | 0.792931 | 0.828364 | 0.807027 | 0.856683 | +0.038276 [+0.006972, +0.077877] | consider logistic |

## Canvas 256
| Feature | Current | Z-score | Geometry | Logistic OOF | Always B3 | Oracle | Logistic-Z Δ 95% CI | Decision hint |
|---|---:|---:|---:|---:|---:|---:|---|---|
| T18 corrected | 0.830202 | 0.827030 | 0.775154 | 0.812495 | 0.804913 | 0.861486 | -0.014535 [-0.046815, +0.012692] | prefer z-score |
| DINO global | 0.805351 | 0.804522 | 0.788073 | 0.805877 | 0.791988 | 0.851277 | +0.001355 [-0.017192, +0.021988] | prefer z-score |
| DINO patch average | 0.806499 | 0.811376 | 0.812705 | 0.821732 | 0.806683 | 0.848212 | +0.010357 [-0.004346, +0.030798] | prefer z-score |
| Target pooling | 0.769089 | 0.764374 | 0.810430 | 0.804951 | 0.792900 | 0.858190 | +0.040576 [+0.005962, +0.080727] | consider logistic |
| Patch correspondence | 0.783248 | 0.784134 | 0.817698 | 0.816068 | 0.828202 | 0.866338 | +0.031934 [-0.002419, +0.068750] | prefer z-score |

## OOF Regret And Closure
| Canvas | Feature | Method | OOF Dice | OOF regret | regret closure vs current | selected D/B1/B2/B3 |
|---|---|---|---:|---:|---:|---:|
| 512 | T18 corrected | current | 0.804241 | 0.037813 | 0.000 | 27/25/14/34 |
| 512 | T18 corrected | route_wise_zscore | 0.806739 | 0.035315 | 0.066 | 19/16/48/17 |
| 512 | T18 corrected | explicit_geometry | 0.791819 | 0.050235 | -0.329 | 0/4/24/72 |
| 512 | T18 corrected | pairwise_logistic | 0.802060 | 0.039994 | -0.058 | 10/28/17/45 |
| 512 | T18 corrected | always_bridge_3 | 0.804409 | 0.037645 | 0.004 | 0/0/0/100 |
| 512 | T18 corrected | oracle | 0.842054 | 0.000000 | 1.000 | 33/22/21/24 |
| 512 | DINO global | current | 0.801313 | 0.047611 | 0.000 | 26/29/21/24 |
| 512 | DINO global | route_wise_zscore | 0.801218 | 0.047707 | -0.002 | 7/18/22/53 |
| 512 | DINO global | explicit_geometry | 0.730830 | 0.118095 | -1.480 | 42/9/22/27 |
| 512 | DINO global | pairwise_logistic | 0.803899 | 0.045025 | 0.054 | 46/11/18/25 |
| 512 | DINO global | always_bridge_3 | 0.767032 | 0.081892 | -0.720 | 0/0/0/100 |
| 512 | DINO global | oracle | 0.848924 | 0.000000 | 1.000 | 26/24/20/30 |
| 512 | DINO patch average | current | 0.777063 | 0.069062 | 0.000 | 34/19/19/28 |
| 512 | DINO patch average | route_wise_zscore | 0.776978 | 0.069147 | -0.001 | 13/45/35/7 |
| 512 | DINO patch average | explicit_geometry | 0.770175 | 0.075950 | -0.100 | 0/5/26/69 |
| 512 | DINO patch average | pairwise_logistic | 0.799450 | 0.046675 | 0.324 | 3/11/17/69 |
| 512 | DINO patch average | always_bridge_3 | 0.782758 | 0.063367 | 0.082 | 0/0/0/100 |
| 512 | DINO patch average | oracle | 0.846125 | 0.000000 | 1.000 | 27/25/17/31 |
| 512 | Target pooling | current | 0.766048 | 0.082691 | 0.000 | 26/23/30/21 |
| 512 | Target pooling | route_wise_zscore | 0.765609 | 0.083131 | -0.005 | 22/46/21/11 |
| 512 | Target pooling | explicit_geometry | 0.800039 | 0.048701 | 0.411 | 0/9/68/23 |
| 512 | Target pooling | pairwise_logistic | 0.799513 | 0.049227 | 0.405 | 69/4/7/20 |
| 512 | Target pooling | always_bridge_3 | 0.770852 | 0.077888 | 0.058 | 0/0/0/100 |
| 512 | Target pooling | oracle | 0.848740 | 0.000000 | 1.000 | 23/18/22/37 |
| 512 | Patch correspondence | current | 0.793123 | 0.063560 | 0.000 | 23/19/30/28 |
| 512 | Patch correspondence | route_wise_zscore | 0.790088 | 0.066595 | -0.048 | 10/69/13/8 |
| 512 | Patch correspondence | explicit_geometry | 0.792931 | 0.063752 | -0.003 | 1/31/45/23 |
| 512 | Patch correspondence | pairwise_logistic | 0.828364 | 0.028319 | 0.554 | 47/3/6/44 |
| 512 | Patch correspondence | always_bridge_3 | 0.807027 | 0.049656 | 0.219 | 0/0/0/100 |
| 512 | Patch correspondence | oracle | 0.856683 | 0.000000 | 1.000 | 30/15/23/32 |
| 256 | T18 corrected | current | 0.830202 | 0.031284 | 0.000 | 20/30/16/34 |
| 256 | T18 corrected | route_wise_zscore | 0.827030 | 0.034456 | -0.101 | 24/13/41/22 |
| 256 | T18 corrected | explicit_geometry | 0.775154 | 0.086332 | -1.760 | 0/20/25/55 |
| 256 | T18 corrected | pairwise_logistic | 0.812495 | 0.048991 | -0.566 | 29/49/15/7 |
| 256 | T18 corrected | always_bridge_3 | 0.804913 | 0.056572 | -0.808 | 0/0/0/100 |
| 256 | T18 corrected | oracle | 0.861486 | 0.000000 | 1.000 | 26/24/19/31 |
| 256 | DINO global | current | 0.805351 | 0.045926 | 0.000 | 17/35/28/20 |
| 256 | DINO global | route_wise_zscore | 0.804522 | 0.046755 | -0.018 | 20/10/19/51 |
| 256 | DINO global | explicit_geometry | 0.788073 | 0.063204 | -0.376 | 2/72/25/1 |
| 256 | DINO global | pairwise_logistic | 0.805877 | 0.045400 | 0.011 | 19/5/9/67 |
| 256 | DINO global | always_bridge_3 | 0.791988 | 0.059289 | -0.291 | 0/0/0/100 |
| 256 | DINO global | oracle | 0.851277 | 0.000000 | 1.000 | 29/16/23/32 |
| 256 | DINO patch average | current | 0.806499 | 0.041713 | 0.000 | 26/20/26/28 |
| 256 | DINO patch average | route_wise_zscore | 0.811376 | 0.036836 | 0.117 | 17/14/42/27 |
| 256 | DINO patch average | explicit_geometry | 0.812705 | 0.035507 | 0.149 | 0/7/40/53 |
| 256 | DINO patch average | pairwise_logistic | 0.821732 | 0.026480 | 0.365 | 7/7/20/66 |
| 256 | DINO patch average | always_bridge_3 | 0.806683 | 0.041529 | 0.004 | 0/0/0/100 |
| 256 | DINO patch average | oracle | 0.848212 | 0.000000 | 1.000 | 26/25/20/29 |
| 256 | Target pooling | current | 0.769089 | 0.089101 | 0.000 | 23/27/21/29 |
| 256 | Target pooling | route_wise_zscore | 0.764374 | 0.093816 | -0.053 | 53/9/33/5 |
| 256 | Target pooling | explicit_geometry | 0.810430 | 0.047759 | 0.464 | 0/9/68/23 |
| 256 | Target pooling | pairwise_logistic | 0.804951 | 0.053239 | 0.402 | 62/10/11/17 |
| 256 | Target pooling | always_bridge_3 | 0.792900 | 0.065290 | 0.267 | 0/0/0/100 |
| 256 | Target pooling | oracle | 0.858190 | 0.000000 | 1.000 | 25/18/25/32 |
| 256 | Patch correspondence | current | 0.783248 | 0.083090 | 0.000 | 22/16/29/33 |
| 256 | Patch correspondence | route_wise_zscore | 0.784134 | 0.082204 | 0.011 | 62/3/10/25 |
| 256 | Patch correspondence | explicit_geometry | 0.817698 | 0.048640 | 0.415 | 0/2/20/78 |
| 256 | Patch correspondence | pairwise_logistic | 0.816068 | 0.050270 | 0.395 | 29/0/6/65 |
| 256 | Patch correspondence | always_bridge_3 | 0.828202 | 0.038137 | 0.541 | 0/0/0/100 |
| 256 | Patch correspondence | oracle | 0.866338 | 0.000000 | 1.000 | 28/14/26/32 |

## Validation Cross-Feature Union Oracle
| Canvas | Combo | union oracle | best single oracle | best single feature | Δ_union | unique hash fraction | decision |
|---|---|---:|---:|---|---:|---:|---|
| 512 | T18+target | 0.887004 | 0.856683 | Patch correspondence | +0.030321 | 1.000 | multi-view route proposal |
| 512 | target+patchcorr | 0.869253 | 0.856683 | Patch correspondence | +0.012570 | 1.000 | multi-view route proposal |
| 512 | T18+target+patchcorr | 0.890528 | 0.856683 | Patch correspondence | +0.033845 | 1.000 | multi-view route proposal |
| 512 | all_features | 0.907047 | 0.856683 | Patch correspondence | +0.050364 | 1.000 | multi-view route proposal |
| 256 | T18+target | 0.892694 | 0.866338 | Patch correspondence | +0.026355 | 1.000 | multi-view route proposal |
| 256 | target+patchcorr | 0.880071 | 0.866338 | Patch correspondence | +0.013733 | 1.000 | multi-view route proposal |
| 256 | T18+target+patchcorr | 0.897057 | 0.866338 | Patch correspondence | +0.030718 | 1.000 | multi-view route proposal |
| 256 | all_features | 0.911628 | 0.866338 | Patch correspondence | +0.045289 | 1.000 | multi-view route proposal |

## Files
- `oof_selector_summary.csv`
- `oof_selector_per_query.csv`
- `oof_selector_fold_scores.csv`
- `paired_bootstrap_selector_dice_diffs.csv`
- `pairwise_logistic_fold_coefficients.csv`
- `pairwise_logistic_coefficient_stability.csv`
- `validation_cross_feature_union_oracle.csv`
- `test_exploratory_cross_feature_union_oracle.csv`
