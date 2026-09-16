# Kvasir PC Adaptive Top-K + Spatial Compactness

SAM3-base@256, frozen original 800/100/100 split and 8 supports; no student. Only validation was used for variant selection.

v0 = Top8 mean; v1 = anchor-area adaptive TopK mean; v2 = v1 - 0.05*(1 - topK bounding-box occupancy).

Anchor K: [18, 33, 16, 12, 13, 33, 2, 8]

Validation selection uses 5-fold target-level out-of-fold Ridge predictions (seed 2026, ridge 1.0). No candidate from a held-out target is in its scorer training fold.

| Variant | PC OOF Dice | TP+PC OOF Dice | PC oracle | Mean fixed b3-b6 |
|---|---:|---:|---:|---:|
| v0 | 0.817329 | 0.842603 | 0.860807 | 0.785440 |
| v1 | 0.821820 | 0.841333 | 0.853959 | 0.797006 |
| v2 | 0.814833 | 0.839675 | 0.865960 | 0.797716 |

Predeclared test gate passed: True. Winner: v1.

Gate: PC OOF improvement >=0.003 and fixed b3-b6 mean drop <=0.005. No test is run if no new variant passes. Test results are reported even if the validation-selected variant degrades.

Anchor lesion area is a scale prior, not target lesion size ground truth. Bounding-box occupancy measures concentration, not anatomical correctness. Changes affect route generation; the original fixed candidate selection gap alone does not establish Top8 as its cause.

## Frozen test

| Variant | PC-only Dice | TP+PC Dice | PC oracle |
|---|---:|---:|---:|
| v0 | 0.861688 | 0.874083 | 0.911715 |
| v1 | 0.867138 | 0.860069 | 0.891881 |
