# Validation OOF Selector Matrix

5-fold grouped by `query_id`. All training statistics/models/geometry parameters are fit on 80 validation queries and evaluated OOF on held-out 20 queries per fold. SAM3 is not rerun.

## Canvas 512
| Feature | Current | Z-score | Geometry | Logistic OOF | Always B3 | Oracle | Logistic-Z Δ 95% CI | Decision hint |
|---|---:|---:|---:|---:|---:|---:|---|---|
| T18 corrected | 0.770012 | 0.770012 | 0.791819 | 0.770012 | 0.804409 | 0.842054 | +0.000000 [+0.000000, +0.000000] | prefer z-score |
| DINO global | 0.776038 | 0.776038 | 0.730830 | 0.776038 | 0.767032 | 0.848924 | +0.000000 [+0.000000, +0.000000] | prefer z-score |
| DINO patch average | 0.758151 | 0.758151 | 0.770175 | 0.758151 | 0.782758 | 0.846125 | +0.000000 [+0.000000, +0.000000] | prefer z-score |
| Target pooling | 0.731848 | 0.731848 | 0.800039 | 0.731848 | 0.770852 | 0.848740 | +0.000000 [+0.000000, +0.000000] | prefer z-score |
| Patch correspondence | 0.753812 | 0.753812 | 0.792931 | 0.753812 | 0.807027 | 0.856683 | +0.000000 [+0.000000, +0.000000] | prefer z-score |

## Canvas 256
| Feature | Current | Z-score | Geometry | Logistic OOF | Always B3 | Oracle | Logistic-Z Δ 95% CI | Decision hint |
|---|---:|---:|---:|---:|---:|---:|---|---|
| T18 corrected | 0.767406 | 0.767406 | 0.775154 | 0.767406 | 0.804913 | 0.861486 | +0.000000 [+0.000000, +0.000000] | prefer z-score |
| DINO global | 0.772543 | 0.772543 | 0.788073 | 0.772543 | 0.791988 | 0.851277 | +0.000000 [+0.000000, +0.000000] | prefer z-score |
| DINO patch average | 0.753238 | 0.753238 | 0.812705 | 0.753238 | 0.806683 | 0.848212 | +0.000000 [+0.000000, +0.000000] | prefer z-score |
| Target pooling | 0.733700 | 0.733700 | 0.810430 | 0.733700 | 0.792900 | 0.858190 | +0.000000 [+0.000000, +0.000000] | prefer z-score |
| Patch correspondence | 0.755816 | 0.755816 | 0.817698 | 0.755816 | 0.828202 | 0.866338 | +0.000000 [+0.000000, +0.000000] | prefer z-score |

## OOF Regret And Closure
| Canvas | Feature | Method | OOF Dice | OOF regret | regret closure vs current | selected D/B1/B2/B3 |
|---|---|---|---:|---:|---:|---:|
| 512 | T18 corrected | current | 0.770012 | 0.072042 | 0.000 | 100/0/0/0 |
| 512 | T18 corrected | route_wise_zscore | 0.770012 | 0.072042 | 0.000 | 100/0/0/0 |
| 512 | T18 corrected | explicit_geometry | 0.791819 | 0.050235 | 0.303 | 0/4/24/72 |
| 512 | T18 corrected | pairwise_logistic | 0.770012 | 0.072042 | 0.000 | 100/0/0/0 |
| 512 | T18 corrected | always_bridge_3 | 0.804409 | 0.037645 | 0.477 | 0/0/0/100 |
| 512 | T18 corrected | oracle | 0.842054 | 0.000000 | 1.000 | 33/22/21/24 |
| 512 | DINO global | current | 0.776038 | 0.072886 | 0.000 | 100/0/0/0 |
| 512 | DINO global | route_wise_zscore | 0.776038 | 0.072886 | 0.000 | 100/0/0/0 |
| 512 | DINO global | explicit_geometry | 0.730830 | 0.118095 | -0.620 | 42/9/22/27 |
| 512 | DINO global | pairwise_logistic | 0.776038 | 0.072886 | 0.000 | 100/0/0/0 |
| 512 | DINO global | always_bridge_3 | 0.767032 | 0.081892 | -0.124 | 0/0/0/100 |
| 512 | DINO global | oracle | 0.848924 | 0.000000 | 1.000 | 26/24/20/30 |
| 512 | DINO patch average | current | 0.758151 | 0.087974 | 0.000 | 100/0/0/0 |
| 512 | DINO patch average | route_wise_zscore | 0.758151 | 0.087974 | 0.000 | 100/0/0/0 |
| 512 | DINO patch average | explicit_geometry | 0.770175 | 0.075950 | 0.137 | 0/5/26/69 |
| 512 | DINO patch average | pairwise_logistic | 0.758151 | 0.087974 | 0.000 | 100/0/0/0 |
| 512 | DINO patch average | always_bridge_3 | 0.782758 | 0.063367 | 0.280 | 0/0/0/100 |
| 512 | DINO patch average | oracle | 0.846125 | 0.000000 | 1.000 | 27/25/17/31 |
| 512 | Target pooling | current | 0.731848 | 0.116892 | 0.000 | 100/0/0/0 |
| 512 | Target pooling | route_wise_zscore | 0.731848 | 0.116892 | 0.000 | 100/0/0/0 |
| 512 | Target pooling | explicit_geometry | 0.800039 | 0.048701 | 0.583 | 0/9/68/23 |
| 512 | Target pooling | pairwise_logistic | 0.731848 | 0.116892 | 0.000 | 100/0/0/0 |
| 512 | Target pooling | always_bridge_3 | 0.770852 | 0.077888 | 0.334 | 0/0/0/100 |
| 512 | Target pooling | oracle | 0.848740 | 0.000000 | 1.000 | 23/18/22/37 |
| 512 | Patch correspondence | current | 0.753812 | 0.102870 | 0.000 | 100/0/0/0 |
| 512 | Patch correspondence | route_wise_zscore | 0.753812 | 0.102870 | 0.000 | 100/0/0/0 |
| 512 | Patch correspondence | explicit_geometry | 0.792931 | 0.063752 | 0.380 | 1/31/45/23 |
| 512 | Patch correspondence | pairwise_logistic | 0.753812 | 0.102870 | 0.000 | 100/0/0/0 |
| 512 | Patch correspondence | always_bridge_3 | 0.807027 | 0.049656 | 0.517 | 0/0/0/100 |
| 512 | Patch correspondence | oracle | 0.856683 | 0.000000 | 1.000 | 30/15/23/32 |
| 256 | T18 corrected | current | 0.767406 | 0.094080 | 0.000 | 100/0/0/0 |
| 256 | T18 corrected | route_wise_zscore | 0.767406 | 0.094080 | 0.000 | 100/0/0/0 |
| 256 | T18 corrected | explicit_geometry | 0.775154 | 0.086332 | 0.082 | 0/20/25/55 |
| 256 | T18 corrected | pairwise_logistic | 0.767406 | 0.094080 | 0.000 | 100/0/0/0 |
| 256 | T18 corrected | always_bridge_3 | 0.804913 | 0.056572 | 0.399 | 0/0/0/100 |
| 256 | T18 corrected | oracle | 0.861486 | 0.000000 | 1.000 | 26/24/19/31 |
| 256 | DINO global | current | 0.772543 | 0.078734 | 0.000 | 100/0/0/0 |
| 256 | DINO global | route_wise_zscore | 0.772543 | 0.078734 | 0.000 | 100/0/0/0 |
| 256 | DINO global | explicit_geometry | 0.788073 | 0.063204 | 0.197 | 2/72/25/1 |
| 256 | DINO global | pairwise_logistic | 0.772543 | 0.078734 | 0.000 | 100/0/0/0 |
| 256 | DINO global | always_bridge_3 | 0.791988 | 0.059289 | 0.247 | 0/0/0/100 |
| 256 | DINO global | oracle | 0.851277 | 0.000000 | 1.000 | 29/16/23/32 |
| 256 | DINO patch average | current | 0.753238 | 0.094974 | 0.000 | 100/0/0/0 |
| 256 | DINO patch average | route_wise_zscore | 0.753238 | 0.094974 | 0.000 | 100/0/0/0 |
| 256 | DINO patch average | explicit_geometry | 0.812705 | 0.035507 | 0.626 | 0/7/40/53 |
| 256 | DINO patch average | pairwise_logistic | 0.753238 | 0.094974 | 0.000 | 100/0/0/0 |
| 256 | DINO patch average | always_bridge_3 | 0.806683 | 0.041529 | 0.563 | 0/0/0/100 |
| 256 | DINO patch average | oracle | 0.848212 | 0.000000 | 1.000 | 26/25/20/29 |
| 256 | Target pooling | current | 0.733700 | 0.124490 | 0.000 | 100/0/0/0 |
| 256 | Target pooling | route_wise_zscore | 0.733700 | 0.124490 | 0.000 | 100/0/0/0 |
| 256 | Target pooling | explicit_geometry | 0.810430 | 0.047759 | 0.616 | 0/9/68/23 |
| 256 | Target pooling | pairwise_logistic | 0.733700 | 0.124490 | 0.000 | 100/0/0/0 |
| 256 | Target pooling | always_bridge_3 | 0.792900 | 0.065290 | 0.476 | 0/0/0/100 |
| 256 | Target pooling | oracle | 0.858190 | 0.000000 | 1.000 | 25/18/25/32 |
| 256 | Patch correspondence | current | 0.755816 | 0.110523 | 0.000 | 100/0/0/0 |
| 256 | Patch correspondence | route_wise_zscore | 0.755816 | 0.110523 | 0.000 | 100/0/0/0 |
| 256 | Patch correspondence | explicit_geometry | 0.817698 | 0.048640 | 0.560 | 0/2/20/78 |
| 256 | Patch correspondence | pairwise_logistic | 0.755816 | 0.110523 | 0.000 | 100/0/0/0 |
| 256 | Patch correspondence | always_bridge_3 | 0.828202 | 0.038137 | 0.655 | 0/0/0/100 |
| 256 | Patch correspondence | oracle | 0.866338 | 0.000000 | 1.000 | 28/14/26/32 |

## Validation Cross-Feature Union Oracle
| Canvas | Combo | union oracle | best single oracle | best single feature | Δ_union | unique hash fraction | decision |
|---|---|---:|---:|---|---:|---:|---|
| 512 | T18+target | 0.887004 | 0.856683 | Patch correspondence | +0.030321 | 1.000 | multi-view useful; hashes mostly unique |
| 512 | target+patchcorr | 0.869253 | 0.856683 | Patch correspondence | +0.012570 | 1.000 | multi-view useful; hashes mostly unique |
| 512 | T18+target+patchcorr | 0.890528 | 0.856683 | Patch correspondence | +0.033845 | 1.000 | multi-view useful; hashes mostly unique |
| 512 | all_features | 0.907047 | 0.856683 | Patch correspondence | +0.050364 | 1.000 | multi-view useful; hashes mostly unique |
| 256 | T18+target | 0.892694 | 0.866338 | Patch correspondence | +0.026355 | 1.000 | multi-view useful; hashes mostly unique |
| 256 | target+patchcorr | 0.880071 | 0.866338 | Patch correspondence | +0.013733 | 1.000 | multi-view useful; hashes mostly unique |
| 256 | T18+target+patchcorr | 0.897057 | 0.866338 | Patch correspondence | +0.030718 | 1.000 | multi-view useful; hashes mostly unique |
| 256 | all_features | 0.911628 | 0.866338 | Patch correspondence | +0.045289 | 1.000 | multi-view useful; hashes mostly unique |

## Files
- `oof_selector_summary.csv`
- `oof_selector_per_query.csv`
- `oof_selector_fold_scores.csv`
- `paired_bootstrap_selector_dice_diffs.csv`
- `pairwise_logistic_fold_coefficients.csv`
- `pairwise_logistic_coefficient_stability.csv`
- `validation_cross_feature_union_oracle.csv`
- `test_exploratory_cross_feature_union_oracle.csv`
