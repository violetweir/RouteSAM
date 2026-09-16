# E1 Test Diagnostics: Analysis Only

No SAM3 rerun. Inputs: existing `route_audit_256_512/all_modes_all_canvas_per_query_routes.csv`.

## 256/512 Oracle Route Consistency
| feature mode | n | same | consistency |
|---|---:|---:|---:|
| t18_corrected | 100 | 59 | 0.590 |
| dino_global_pooling | 100 | 61 | 0.610 |
| dino_patch_average | 100 | 63 | 0.630 |
| anchor_conditioned_target_pooling | 100 | 67 | 0.670 |
| anchor_conditioned_patch_correspondence | 100 | 57 | 0.570 |

## 256/512 Route Dice Ranking Correlation
| feature mode | mean per-query Spearman | flat route Dice Spearman |
|---|---:|---:|
| t18_corrected | 0.555 | 0.879 |
| dino_global_pooling | 0.633 | 0.966 |
| dino_patch_average | 0.659 | 0.926 |
| anchor_conditioned_target_pooling | 0.601 | 0.937 |
| anchor_conditioned_patch_correspondence | 0.561 | 0.949 |

## Cross-Feature Union Oracle
| canvas | union scope | oracle Dice |
|---|---|---:|
| 512 | direct_only | 0.840944 |
| 512 | direct_b1 | 0.873195 |
| 512 | direct_b1_b2 | 0.896489 |
| 512 | direct_b1_b2_b3 | 0.921158 |
| 256 | direct_only | 0.827421 |
| 256 | direct_b1 | 0.873859 |
| 256 | direct_b1_b2 | 0.903014 |
| 256 | direct_b1_b2_b3 | 0.921385 |

## Score-Dice Spearman, pooled over 4 routes
| canvas | feature mode | KNN bottleneck | KNN mean | q_return |
|---|---|---:|---:|---:|
| 512 | t18_corrected | -0.058 | -0.053 | 0.224 |
| 512 | dino_global_pooling | -0.022 | -0.013 | 0.236 |
| 512 | dino_patch_average | 0.003 | 0.022 | 0.226 |
| 512 | anchor_conditioned_target_pooling | -0.012 | 0.051 | 0.251 |
| 512 | anchor_conditioned_patch_correspondence | 0.182 | 0.121 | 0.331 |
| 256 | t18_corrected | -0.017 | -0.019 | 0.131 |
| 256 | dino_global_pooling | -0.014 | -0.000 | 0.178 |
| 256 | dino_patch_average | 0.015 | 0.036 | 0.101 |
| 256 | anchor_conditioned_target_pooling | -0.019 | 0.059 | 0.233 |
| 256 | anchor_conditioned_patch_correspondence | 0.171 | 0.140 | 0.259 |

## Files
- `oracle_route_consistency_256_512.csv`
- `route_dice_ranking_correlation_256_512.csv`
- `score_dice_spearman.csv`
- `pairwise_route_ranking_accuracy.csv`
- `oracle_selected_regret_matrix.csv`
- `cross_feature_union_oracle.csv`
- `cross_feature_node_jaccard_hash_overlap.csv`
- `E1_test_diagnostics.json`
