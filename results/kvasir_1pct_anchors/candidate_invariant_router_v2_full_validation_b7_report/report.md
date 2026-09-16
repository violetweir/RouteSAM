# Candidate-Invariant Router v2 Summary

## Per-Feature Test Metrics

| Feature | Best fixed | Fixed Dice | Selected C7 | Oracle C7 | Dynamic gain | Regret | Spearman | Kendall | Fail AUROC | C3->C7 change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | b4 | 0.821736 | 0.845581 | 0.897202 | +0.023845 | 0.051621 | 0.401 | 0.276 | 0.956 | 0.730 |
| dino_global_pooling | b6 | 0.818565 | 0.791717 | 0.889342 | -0.026848 | 0.097625 | 0.318 | 0.212 | 0.869 | 0.060 |
| dino_patch_average | b5 | 0.829179 | 0.815340 | 0.893806 | -0.013839 | 0.078466 | 0.422 | 0.291 | 0.925 | 0.720 |
| anchor_conditioned_target_pooling | b6 | 0.854627 | 0.854437 | 0.898969 | -0.000189 | 0.044531 | 0.449 | 0.316 | 0.953 | 1.000 |
| anchor_conditioned_patch_correspondence | b6 | 0.842078 | 0.851690 | 0.887997 | +0.009613 | 0.036307 | 0.501 | 0.356 | 0.952 | 0.960 |

## Unified Candidate-Pool Oracle

| Modes | Oracle Dice | Mean unique candidates | Range | Oracle histogram |
|---|---:|---:|---:|---|
| anchor_conditioned_target_pooling+anchor_conditioned_patch_correspondence | 0.903904 | 13.5 | 9-16 | `{"anchor_conditioned_patch_correspondence:bridge_1": 12, "anchor_conditioned_patch_correspondence:bridge_2": 7, "anchor_conditioned_patch_correspondence:bridge_3": 12, "anchor_conditioned_patch_correspondence:bridge_4": 8, "anchor_conditioned_patch_correspondence:bridge_5": 3, "anchor_conditioned_patch_correspondence:bridge_6": 7, "anchor_conditioned_patch_correspondence:bridge_7": 2, "anchor_conditioned_patch_correspondence:direct": 10, "anchor_conditioned_target_pooling:bridge_1": 1, "anchor_conditioned_target_pooling:bridge_2": 8, "anchor_conditioned_target_pooling:bridge_3": 4, "anchor_conditioned_target_pooling:bridge_4": 5, "anchor_conditioned_target_pooling:bridge_5": 7, "anchor_conditioned_target_pooling:bridge_6": 13, "anchor_conditioned_target_pooling:direct": 1}` |
| anchor_conditioned_target_pooling+anchor_conditioned_patch_correspondence+t18_corrected | 0.923511 | 21.2 | 17-24 | `{"anchor_conditioned_patch_correspondence:bridge_1": 7, "anchor_conditioned_patch_correspondence:bridge_2": 4, "anchor_conditioned_patch_correspondence:bridge_3": 8, "anchor_conditioned_patch_correspondence:bridge_4": 3, "anchor_conditioned_patch_correspondence:bridge_5": 2, "anchor_conditioned_patch_correspondence:bridge_6": 2, "anchor_conditioned_patch_correspondence:bridge_7": 1, "anchor_conditioned_patch_correspondence:direct": 3, "anchor_conditioned_target_pooling:bridge_2": 6, "anchor_conditioned_target_pooling:bridge_3": 2, "anchor_conditioned_target_pooling:bridge_4": 4, "anchor_conditioned_target_pooling:bridge_5": 4, "anchor_conditioned_target_pooling:bridge_6": 9, "anchor_conditioned_target_pooling:direct": 1, "t18_corrected:bridge_1": 4, "t18_corrected:bridge_2": 5, "t18_corrected:bridge_3": 7, "t18_corrected:bridge_4": 4, "t18_corrected:bridge_5": 4, "t18_corrected:bridge_6": 6, "t18_corrected:bridge_7": 3, "t18_corrected:direct": 11}` |

