# OOF Decision Summary

This summary uses validation OOF only. Logistic is treated as supervised diagnostic unless calibration labels are explicitly budgeted.

## Selector Matrix Decision
| Canvas | Feature | Current | Z-score | Geometry | Logistic | Always B3 | Oracle | Log-Z Δ 95% CI | fold signs | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| 512 | T18 corrected | 0.804241 | 0.806739 | 0.791819 | 0.802060 | 0.804409 | 0.842054 | -0.004678 [-0.012300,+0.000631] | 1/5 | prefer_zscore |
| 512 | DINO global | 0.801313 | 0.801218 | 0.730830 | 0.803899 | 0.767032 | 0.848924 | +0.002682 [-0.016105,+0.022644] | 3/5 | prefer_zscore |
| 512 | DINO patch average | 0.777063 | 0.776978 | 0.770175 | 0.799450 | 0.782758 | 0.846125 | +0.022471 [-0.002593,+0.055674] | 5/5 | logistic_positive_but_ci_uncertain |
| 512 | Target pooling | 0.766048 | 0.765609 | 0.800039 | 0.799513 | 0.770852 | 0.848740 | +0.033904 [+0.001172,+0.072722] | 4/5 | logistic_supervised_diagnostic_strong |
| 512 | Patch correspondence | 0.793123 | 0.790088 | 0.792931 | 0.828364 | 0.807027 | 0.856683 | +0.038276 [+0.006972,+0.077877] | 4/5 | logistic_supervised_diagnostic_strong |
| 256 | T18 corrected | 0.830202 | 0.827030 | 0.775154 | 0.812495 | 0.804913 | 0.861486 | -0.014535 [-0.046815,+0.012692] | 4/5 | prefer_zscore |
| 256 | DINO global | 0.805351 | 0.804522 | 0.788073 | 0.805877 | 0.791988 | 0.851277 | +0.001355 [-0.017192,+0.021988] | 2/5 | prefer_zscore |
| 256 | DINO patch average | 0.806499 | 0.811376 | 0.812705 | 0.821732 | 0.806683 | 0.848212 | +0.010357 [-0.004346,+0.030798] | 3/5 | prefer_zscore |
| 256 | Target pooling | 0.769089 | 0.764374 | 0.810430 | 0.804951 | 0.792900 | 0.858190 | +0.040576 [+0.005962,+0.080727] | 4/5 | logistic_supervised_diagnostic_strong |
| 256 | Patch correspondence | 0.783248 | 0.784134 | 0.817698 | 0.816068 | 0.828202 | 0.866338 | +0.031934 [-0.002419,+0.068750] | 4/5 | logistic_positive_but_ci_uncertain |

## Best Feature Per Selector
| Canvas | Selector | best feature | OOF Dice | OOF regret |
|---|---|---|---:|---:|
| 512 | current | T18 corrected | 0.804241 | 0.037813 |
| 512 | route_wise_zscore | T18 corrected | 0.806739 | 0.035315 |
| 512 | explicit_geometry | Target pooling | 0.800039 | 0.048701 |
| 512 | pairwise_logistic | Patch correspondence | 0.828364 | 0.028319 |
| 512 | always_bridge_3 | Patch correspondence | 0.807027 | 0.049656 |
| 512 | oracle | Patch correspondence | 0.856683 | 0.000000 |
| 256 | current | T18 corrected | 0.830202 | 0.031284 |
| 256 | route_wise_zscore | T18 corrected | 0.827030 | 0.034456 |
| 256 | explicit_geometry | Patch correspondence | 0.817698 | 0.048640 |
| 256 | pairwise_logistic | DINO patch average | 0.821732 | 0.026480 |
| 256 | always_bridge_3 | Patch correspondence | 0.828202 | 0.038137 |
| 256 | oracle | Patch correspondence | 0.866338 | 0.000000 |

## Stable Logistic Coefficient Signs
| Canvas | Feature mode | coefficient | mean | std | +folds | -folds | sign consistency |
|---|---|---|---:|---:|---:|---:|---:|
| 512 | T18 corrected | q_return | +0.1825 | 0.0945 | 5 | 0 | 1.00 |
| 512 | T18 corrected | q_multi | +0.4341 | 0.1161 | 5 | 0 | 1.00 |
| 512 | T18 corrected | knn_mean | +0.1343 | 0.0863 | 4 | 1 | 0.80 |
| 512 | T18 corrected | q_return*q_multi | -0.0445 | 0.1461 | 1 | 4 | 0.80 |
| 512 | T18 corrected | q_return-q_multi | +0.0480 | 0.0798 | 4 | 1 | 0.80 |
| 512 | DINO global | q_return | +0.3376 | 0.0871 | 5 | 0 | 1.00 |
| 512 | DINO global | q_multi | +1.1290 | 0.1142 | 5 | 0 | 1.00 |
| 512 | DINO global | knn_bottleneck | +0.2953 | 0.1285 | 5 | 0 | 1.00 |
| 512 | DINO global | knn_mean | -0.5407 | 0.1448 | 0 | 5 | 1.00 |
| 512 | DINO global | length | +0.1033 | 0.0384 | 5 | 0 | 1.00 |
| 512 | DINO global | q_return*q_multi | -0.4821 | 0.1600 | 0 | 5 | 1.00 |
| 512 | DINO global | q_return-q_multi | -0.1167 | 0.0790 | 1 | 4 | 0.80 |
| 512 | DINO patch average | q_return | +0.4565 | 0.0771 | 5 | 0 | 1.00 |
| 512 | DINO patch average | q_multi | +0.8109 | 0.1232 | 5 | 0 | 1.00 |
| 512 | DINO patch average | knn_bottleneck | -0.4242 | 0.1408 | 0 | 5 | 1.00 |
| 512 | DINO patch average | knn_mean | +0.4145 | 0.1660 | 5 | 0 | 1.00 |
| 512 | DINO patch average | length | +0.0468 | 0.0292 | 4 | 1 | 0.80 |
| 512 | DINO patch average | q_return*q_multi | -0.7155 | 0.1377 | 0 | 5 | 1.00 |
| 512 | DINO patch average | q_return-q_multi | +0.1182 | 0.0691 | 5 | 0 | 1.00 |
| 512 | Target pooling | q_return | -0.0521 | 0.0406 | 1 | 4 | 0.80 |
| 512 | Target pooling | q_multi | +0.8880 | 0.1001 | 5 | 0 | 1.00 |
| 512 | Target pooling | knn_bottleneck | +0.1408 | 0.1158 | 4 | 1 | 0.80 |
| 512 | Target pooling | knn_mean | -0.2424 | 0.1016 | 0 | 5 | 1.00 |
| 512 | Target pooling | length | +0.0861 | 0.0829 | 4 | 1 | 0.80 |
| 512 | Target pooling | q_return-q_multi | -0.3763 | 0.0567 | 0 | 5 | 1.00 |
| 512 | Patch correspondence | q_multi | +0.5909 | 0.1012 | 5 | 0 | 1.00 |
| 512 | Patch correspondence | knn_bottleneck | +0.1622 | 0.2013 | 4 | 1 | 0.80 |
| 512 | Patch correspondence | knn_mean | -0.2254 | 0.0860 | 0 | 5 | 1.00 |
| 512 | Patch correspondence | length | +0.1315 | 0.0730 | 5 | 0 | 1.00 |
| 512 | Patch correspondence | q_return-q_multi | -0.2193 | 0.0549 | 0 | 5 | 1.00 |
| 256 | T18 corrected | q_return | +0.3233 | 0.1676 | 5 | 0 | 1.00 |
| 256 | T18 corrected | q_multi | +0.8277 | 0.1645 | 5 | 0 | 1.00 |
| 256 | T18 corrected | knn_mean | +0.1357 | 0.1382 | 4 | 1 | 0.80 |
| 256 | T18 corrected | length | -0.0584 | 0.0187 | 0 | 5 | 1.00 |
| 256 | T18 corrected | q_return*q_multi | -0.2558 | 0.2890 | 1 | 4 | 0.80 |
| 256 | T18 corrected | q_return-q_multi | +0.0776 | 0.1384 | 4 | 1 | 0.80 |
| 256 | DINO global | q_return | +0.5033 | 0.0948 | 5 | 0 | 1.00 |
| 256 | DINO global | q_multi | +0.6634 | 0.1678 | 5 | 0 | 1.00 |
| 256 | DINO global | knn_bottleneck | +0.2018 | 0.0704 | 5 | 0 | 1.00 |
| 256 | DINO global | knn_mean | -0.3882 | 0.0758 | 0 | 5 | 1.00 |
| 256 | DINO global | length | +0.1777 | 0.0515 | 5 | 0 | 1.00 |
| 256 | DINO global | q_return*q_multi | -0.6862 | 0.1165 | 0 | 5 | 1.00 |
| 256 | DINO global | q_return-q_multi | +0.2960 | 0.0590 | 5 | 0 | 1.00 |
| 256 | DINO patch average | q_return | +0.2808 | 0.0401 | 5 | 0 | 1.00 |
| 256 | DINO patch average | q_multi | +0.6362 | 0.0589 | 5 | 0 | 1.00 |
| 256 | DINO patch average | knn_bottleneck | +0.0907 | 0.1148 | 4 | 1 | 0.80 |
| 256 | DINO patch average | knn_mean | -0.0857 | 0.1745 | 1 | 4 | 0.80 |
| 256 | DINO patch average | length | +0.0541 | 0.0356 | 5 | 0 | 1.00 |
| 256 | DINO patch average | q_return*q_multi | -0.4469 | 0.1632 | 0 | 5 | 1.00 |
| 256 | Target pooling | q_return | -0.0431 | 0.0452 | 1 | 4 | 0.80 |
| 256 | Target pooling | q_multi | +0.4625 | 0.1608 | 5 | 0 | 1.00 |
| 256 | Target pooling | knn_bottleneck | -0.0377 | 0.1859 | 4 | 1 | 0.80 |
| 256 | Target pooling | knn_mean | -0.1435 | 0.1006 | 0 | 5 | 1.00 |
| 256 | Target pooling | q_return-q_multi | -0.1980 | 0.0355 | 0 | 5 | 1.00 |
| 256 | Patch correspondence | q_return | +0.0435 | 0.1678 | 4 | 1 | 0.80 |
| 256 | Patch correspondence | q_multi | +0.4465 | 0.1926 | 5 | 0 | 1.00 |
| 256 | Patch correspondence | knn_bottleneck | +0.1092 | 0.0337 | 5 | 0 | 1.00 |
| 256 | Patch correspondence | knn_mean | -0.3239 | 0.0969 | 0 | 5 | 1.00 |
| 256 | Patch correspondence | length | +0.2440 | 0.0700 | 5 | 0 | 1.00 |
| 256 | Patch correspondence | q_return*q_multi | -0.0946 | 0.2175 | 1 | 4 | 0.80 |
| 256 | Patch correspondence | q_return-q_multi | -0.1416 | 0.0660 | 0 | 5 | 1.00 |

## Union Oracle Decision
| Canvas | Combo | Delta union | unique fraction | duplicate queries | decision |
|---|---|---:|---:|---:|---|
| 512 | T18+target | +0.030321 | 0.974 | 21 | multi-view proposal recommended |
| 512 | target+patchcorr | +0.012570 | 0.780 | 70 | multi-view proposal recommended |
| 512 | T18+target+patchcorr | +0.033845 | 0.833 | 78 | multi-view proposal recommended |
| 512 | all_features | +0.050364 | 0.782 | 99 | multi-view proposal recommended |
| 256 | T18+target | +0.026355 | 0.974 | 21 | multi-view proposal recommended |
| 256 | target+patchcorr | +0.013733 | 0.780 | 70 | multi-view proposal recommended |
| 256 | T18+target+patchcorr | +0.030718 | 0.833 | 78 | multi-view proposal recommended |
| 256 | all_features | +0.045289 | 0.782 | 99 | multi-view proposal recommended |
